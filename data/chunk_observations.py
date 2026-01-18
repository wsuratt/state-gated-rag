"""
Chunk observations into retrievable segments for state-gated RAG.

WebShop observations use [SEP] as delimiters between elements:
- Search page: instruction, search box
- Results page: instruction, navigation, products (ASIN, name, price)
- Product page: instruction, navigation, options (colors, sizes), product details, Buy Now
"""

import re
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Chunk:
    """A chunk of observation text with metadata."""
    text: str
    chunk_type: str  # 'instruction', 'navigation', 'product', 'option', 'action', 'misc'
    start_idx: int   # Position in original observation


def chunk_webshop_observation(
    obs: str,
    max_chunk_len: int = 800,
    min_chunk_len: int = 10,
) -> List[str]:
    """
    Split a WebShop observation into retrievable chunks.

    Strategy:
    1. Split by [SEP] delimiter
    2. Classify and group related segments (products, options)
    3. Enforce max/min length constraints

    Args:
        obs: Raw observation text from WebShop
        max_chunk_len: Maximum characters per chunk
        min_chunk_len: Minimum characters to keep a chunk

    Returns:
        List of chunk strings
    """
    # Split by [SEP] delimiter
    segments = [s.strip() for s in obs.split('[SEP]')]
    segments = [s for s in segments if s]  # Remove empty

    chunks = []

    # First pass: identify option groups (look ahead)
    option_groups = _find_option_groups(segments)

    # Classify and process segments
    i = 0
    skip_until = -1  # For skipping segments already processed

    while i < len(segments):
        # Skip if we've already processed this segment
        if i < skip_until:
            i += 1
            continue

        segment = segments[i]

        # Check if this is part of an option group FIRST (before length check)
        # because option headers like "color", "size" are short
        if i in option_groups:
            option_chunk, end_idx = option_groups[i]
            chunks.append(option_chunk)
            skip_until = end_idx
            i += 1
            continue

        # Skip very short segments (likely UI cruft like "Instruction:")
        if len(segment) < min_chunk_len or segment.lower() in ['instruction:', 'webshop']:
            i += 1
            continue

        # Check if this is a product listing (starts with ASIN pattern)
        if _is_asin(segment):
            # Group ASIN + product name + price into one chunk
            product_chunk, consumed = _extract_product_chunk(segments, i)
            if product_chunk:
                chunks.append(product_chunk)
            i += consumed
            continue

        # Check for navigation elements - group them
        if _is_navigation(segment):
            nav_chunk, consumed = _extract_navigation_chunk(segments, i)
            chunks.append(nav_chunk)
            i += consumed
            continue

        # Check for action buttons
        if _is_action_button(segment):
            chunks.append(f"[Action] {segment}")
            i += 1
            continue

        # Default: add as content chunk
        if len(segment) >= min_chunk_len:
            chunks.append(segment)
        i += 1

    # Post-process: enforce max length
    final_chunks = []
    for chunk in chunks:
        if len(chunk) <= max_chunk_len:
            final_chunks.append(chunk)
        else:
            # Split long chunks
            final_chunks.extend(_split_long_chunk(chunk, max_chunk_len))

    return final_chunks


def _find_option_groups(segments: List[str]) -> dict:
    """
    Pre-scan segments to find option groups (header + values).

    Returns:
        Dict mapping start_idx -> (chunk_text, end_idx)
    """
    groups = {}
    i = 0

    while i < len(segments):
        segment = segments[i].strip().lower()

        if _is_option_header(segments[i]):
            header = segments[i].strip()
            options = []
            j = i + 1

            # Collect option values
            while j < len(segments):
                opt = segments[j].strip()

                # Stop conditions
                if _is_option_header(opt):
                    break
                if _is_asin(opt):
                    break
                if _is_action_button(opt):
                    break
                if _is_price(opt) or opt.lower().startswith('price:'):
                    break
                if opt.lower().startswith('rating:'):
                    break
                if len(opt) > 40:  # Likely product description
                    break
                if opt.lower() in ['instruction:', 'webshop']:
                    break
                # Options are typically 1-3 words
                if len(opt.split()) > 4:
                    break

                options.append(opt)
                j += 1

            if options:
                chunk_text = f"[Options] {header}: {', '.join(options)}"
                groups[i] = (chunk_text, j)

        i += 1

    return groups


def _extract_navigation_chunk(segments: List[str], start_idx: int) -> Tuple[str, int]:
    """Extract consecutive navigation elements into one chunk."""
    nav_items = []
    i = start_idx

    while i < len(segments):
        segment = segments[i].strip()
        if _is_navigation(segment):
            nav_items.append(segment)
            i += 1
        else:
            break

    return f"[Navigation] {' | '.join(nav_items)}", len(nav_items)


def _is_asin(text: str) -> bool:
    """Check if text looks like an Amazon ASIN (B0XXXXXXXXX pattern)."""
    return bool(re.match(r'^B0[A-Z0-9]{8,10}$', text.strip()))


def _extract_product_chunk(segments: List[str], start_idx: int) -> Tuple[Optional[str], int]:
    """
    Extract a product chunk: ASIN + name + price.

    Returns:
        (chunk_text, num_segments_consumed)
    """
    if start_idx >= len(segments):
        return None, 1

    asin = segments[start_idx]
    name = segments[start_idx + 1] if start_idx + 1 < len(segments) else ""
    consumed = 2

    # Check if next segment looks like a price
    if start_idx + 2 < len(segments):
        potential_price = segments[start_idx + 2]
        if _is_price(potential_price):
            consumed = 3
            return f"[Product] {asin}: {name} - {potential_price}", consumed

    # Just ASIN and name
    return f"[Product] {asin}: {name}", consumed


def _is_price(text: str) -> bool:
    """Check if text looks like a price."""
    return bool(re.match(r'^\$[\d.,]+(\s*to\s*\$[\d.,]+)?$', text.strip()))


def _is_option_header(text: str) -> bool:
    """Check if text is an option group header (color, size, etc.)."""
    headers = ['color', 'size', 'style', 'pattern', 'material', 'count', 'flavor', 'scent']
    return text.strip().lower() in headers


def _extract_option_chunk(segments: List[str], start_idx: int) -> Tuple[Optional[str], int]:
    """
    Extract an option group: header + all option values.

    Returns:
        (chunk_text, num_segments_consumed)
    """
    header = segments[start_idx]
    options = []
    consumed = 1

    # Collect option values until we hit another header, ASIN, or action
    for i in range(start_idx + 1, len(segments)):
        segment = segments[i].strip()

        # Stop conditions
        if _is_option_header(segment):
            break
        if _is_asin(segment):
            break
        if _is_action_button(segment):
            break
        if _is_navigation(segment):
            break
        if len(segment) > 50:  # Likely product description, not option
            break
        if _is_price(segment):
            break

        options.append(segment)
        consumed += 1

    if options:
        return f"[Options] {header}: {', '.join(options)}", consumed
    return None, 1


def _is_navigation(text: str) -> bool:
    """Check if text is a navigation element."""
    nav_patterns = [
        r'^Back to Search$',
        r'^< Prev$',
        r'^Next >$',
        r'^Page \d+',
        r'^\(Total results: \d+\)$',
    ]
    text = text.strip()
    return any(re.match(p, text, re.IGNORECASE) for p in nav_patterns)


def _is_action_button(text: str) -> bool:
    """Check if text is an action button."""
    actions = ['buy now', 'add to cart', 'description', 'features', 'reviews', 'search']
    return text.strip().lower() in actions


def _split_long_chunk(text: str, max_len: int) -> List[str]:
    """Split a long chunk into smaller pieces."""
    chunks = []

    # Try to split on sentence boundaries first
    sentences = re.split(r'(?<=[.!?])\s+', text)

    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= max_len:
            current_chunk = (current_chunk + " " + sentence).strip()
        else:
            if current_chunk:
                chunks.append(current_chunk)
            # If single sentence is too long, split by words
            if len(sentence) > max_len:
                words = sentence.split()
                current_chunk = ""
                for word in words:
                    if len(current_chunk) + len(word) + 1 <= max_len:
                        current_chunk = (current_chunk + " " + word).strip()
                    else:
                        if current_chunk:
                            chunks.append(current_chunk)
                        current_chunk = word
            else:
                current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def chunk_observation(obs: str, env: str = "webshop", **kwargs) -> List[str]:
    """
    Unified interface for chunking observations from any environment.

    Args:
        obs: Raw observation text
        env: Environment name ('webshop', 'alfworld')
        **kwargs: Additional chunking parameters

    Returns:
        List of chunk strings
    """
    if env == "webshop":
        return chunk_webshop_observation(obs, **kwargs)
    elif env == "alfworld":
        # TODO: Implement ALFWorld chunking
        return chunk_alfworld_observation(obs, **kwargs)
    else:
        raise ValueError(f"Unknown environment: {env}")


def chunk_alfworld_observation(
    obs: str,
    max_chunk_len: int = 800,
    min_chunk_len: int = 10,
) -> List[str]:
    """
    Chunk ALFWorld observations.

    ALFWorld observations are simpler - typically room descriptions
    with objects and receptacles listed.
    """
    # Split by newlines or sentences
    lines = [line.strip() for line in obs.split('\n') if line.strip()]

    chunks = []
    for line in lines:
        if len(line) >= min_chunk_len:
            if len(line) <= max_chunk_len:
                chunks.append(line)
            else:
                chunks.extend(_split_long_chunk(line, max_chunk_len))

    return chunks


# ============================================================================
# Testing / Demo
# ============================================================================

if __name__ == "__main__":
    # Test with sample WebShop observations

    # Search results page
    search_obs = """Instruction: [SEP] Find me slim fit women's jumpsuits [SEP] Back to Search [SEP] Page 1 (Total results: 50) [SEP] Next > [SEP] B099WX3CV5 [SEP] Women Aesthetic Short Sleeve Jumpsuit Bodycon Sexy V Neck [SEP] $13.99 to $24.89 [SEP] B09QCVCYVY [SEP] High Waist Bike Shorts Tummy Control [SEP] $6.82"""

    print("=== Search Results Page ===")
    chunks = chunk_webshop_observation(search_obs)
    for i, chunk in enumerate(chunks):
        print(f"{i+1}. {chunk[:80]}..." if len(chunk) > 80 else f"{i+1}. {chunk}")

    print()

    # Product page with options
    product_obs = """Instruction: [SEP] Find me slim fit women's jumpsuits [SEP] Back to Search [SEP] < Prev [SEP] color [SEP] green stripe [SEP] letter blue [SEP] floral purple [SEP] size [SEP] small [SEP] medium [SEP] large [SEP] Women Aesthetic Short Sleeve Jumpsuit [SEP] Price: $13.99 to $24.89 [SEP] Rating: N.A. [SEP] Description [SEP] Features [SEP] Reviews [SEP] Buy Now"""

    print("=== Product Page ===")
    chunks = chunk_webshop_observation(product_obs)
    for i, chunk in enumerate(chunks):
        print(f"{i+1}. {chunk[:80]}..." if len(chunk) > 80 else f"{i+1}. {chunk}")
