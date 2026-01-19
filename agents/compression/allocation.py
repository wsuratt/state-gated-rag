"""
Budget allocation strategies for state-gated compression.

Key insight: At tight budgets, "fair" allocation (everyone gets some budget)
destroys information. Instead, use a triage policy:
- Front buffer: preserve beginning of observation (WebShop is front-loaded)
- Structure anchors: preserve headers, buttons, options
- Hard top-k: give most remaining budget to top-ranked chunks
"""

import re
import numpy as np
from typing import List, Dict, Tuple, Optional
import torch
import torch.nn.functional as F


# Budget thresholds for switching strategies
LOW_BUDGET_THRESHOLD = 768
VERY_LOW_BUDGET_THRESHOLD = 384

# Allocation fractions for low-budget mode
FRONT_BUFFER_FRAC = 0.35  # 35% for front of observation
ANCHOR_FRAC = 0.15  # 15% for structure anchors
TOPK_FRAC = 0.50  # 50% for top-k chunks

# Number of top chunks to give detail at different budgets
TOPK_VERY_LOW = 2
TOPK_LOW = 3
TOPK_MEDIUM = 5


def extract_anchors(observation: str, max_chars: int = 150) -> str:
    """
    Extract structural anchors from WebShop observation.

    Anchors are critical structural elements that should always be preserved:
    - Navigation: Back to Search, Next, Prev
    - Actions: Buy Now, Add to Cart, Description, Features
    - Option labels: Color, Size, Price
    - Product IDs: B0XXXXXXX patterns

    Args:
        observation: Raw observation text
        max_chars: Maximum characters for anchors

    Returns:
        Condensed anchor string
    """
    anchors = []

    # Navigation anchors
    nav_patterns = [
        r'Back to Search',
        r'< Prev',
        r'Next >',
        r'Page \d+',
    ]
    for pattern in nav_patterns:
        match = re.search(pattern, observation)
        if match:
            anchors.append(match.group())

    # Action anchors
    action_patterns = [
        r'Buy Now',
        r'Add to Cart',
        r'Description',
        r'Features',
        r'Reviews',
    ]
    for pattern in action_patterns:
        if pattern in observation:
            anchors.append(f"[{pattern}]")

    # Option labels with their values (condensed)
    size_match = re.search(r'\[SEP\] size \[SEP\](.+?)(?:\[SEP\] color|\[SEP\] [A-Z]|$)', observation)
    if size_match:
        sizes = re.findall(r'\[SEP\] ([a-z0-9x-]+(?:-large|-small)?)', size_match.group(1))
        if sizes:
            anchors.append(f"Size: {', '.join(sizes[:5])}")

    color_match = re.search(r'\[SEP\] color \[SEP\](.+?)(?:\[SEP\] [A-Z]|$)', observation)
    if color_match:
        colors = re.findall(r'\[SEP\] ([a-z ]+)', color_match.group(1))
        if colors:
            anchors.append(f"Color: {', '.join(colors[:5])}")

    # Price
    price_match = re.search(r'Price: \$[\d.]+(?:\s*to\s*\$[\d.]+)?', observation)
    if price_match:
        anchors.append(price_match.group())

    # Product IDs (first 3)
    product_ids = re.findall(r'B[A-Z0-9]{9,10}', observation)
    if product_ids:
        anchors.append(f"Products: {', '.join(product_ids[:3])}")

    # Join and truncate
    result = " | ".join(anchors)
    return result[:max_chars]


def allocate_budget(
    scores: torch.Tensor,
    chunks: List[str],
    total_budget: int,
    mode: str = "auto",
    temperature: float = 1.0,
    min_chunk_budget: int = 30,
    reserve_for_structure: int = 0,
) -> List[int]:
    """
    Allocate budget to chunks based on importance scores.

    Args:
        scores: Importance scores for each chunk (from retriever)
        chunks: List of chunk strings
        total_budget: Total character budget
        mode: Allocation mode:
            - "auto": Choose based on budget size
            - "proportional": Softmax proportional (original)
            - "low_budget_topk": Front buffer + anchors + hard top-k
        temperature: Softmax temperature for proportional mode
        min_chunk_budget: Minimum chars per chunk
        reserve_for_structure: Chars to reserve for front buffer + anchors

    Returns:
        List of budget allocations per chunk
    """
    n_chunks = len(chunks)

    if n_chunks == 0:
        return []

    if n_chunks == 1:
        return [max(total_budget - reserve_for_structure, min_chunk_budget)]

    # Subtract reserved space from available budget
    available_budget = max(total_budget - reserve_for_structure, total_budget // 2)

    # Auto-select mode based on budget
    if mode == "auto":
        if total_budget <= VERY_LOW_BUDGET_THRESHOLD:
            mode = "low_budget_topk"
        elif total_budget <= LOW_BUDGET_THRESHOLD:
            mode = "low_budget_topk"
        else:
            mode = "proportional"

    if mode == "proportional":
        return _allocate_proportional(scores, n_chunks, available_budget, temperature, min_chunk_budget)
    elif mode == "low_budget_topk":
        return _allocate_low_budget(scores, chunks, available_budget, min_chunk_budget)
    else:
        raise ValueError(f"Unknown allocation mode: {mode}")


def _allocate_proportional(
    scores: torch.Tensor,
    n_chunks: int,
    total_budget: int,
    temperature: float,
    min_chunk_budget: int,
) -> List[int]:
    """Original proportional allocation via softmax."""
    budget_proportions = F.softmax(scores / temperature, dim=0).cpu().numpy()
    budgets = (budget_proportions * total_budget).astype(int)
    budgets = [max(int(b), min_chunk_budget) for b in budgets]
    return budgets


def _allocate_low_budget(
    scores: torch.Tensor,
    chunks: List[str],
    total_budget: int,
    min_chunk_budget: int,
) -> List[int]:
    """
    Low-budget allocation with front buffer priority.

    Strategy:
    1. Give first chunk (usually instruction/header) a large share
    2. Give top-k scored chunks the bulk of remaining budget
    3. Give other chunks minimal budget (just headers)
    """
    n_chunks = len(chunks)
    scores_np = scores.cpu().numpy()

    # Determine top-k based on budget
    if total_budget <= VERY_LOW_BUDGET_THRESHOLD:
        top_k = TOPK_VERY_LOW
    else:
        top_k = TOPK_LOW

    top_k = min(top_k, n_chunks)

    # Get top-k indices by score
    sorted_indices = scores_np.argsort()[::-1]
    topk_indices = set(sorted_indices[:top_k])

    # Always include chunk 0 (usually instruction) in top-k
    topk_indices.add(0)

    # Calculate budgets
    # Header budget for non-top-k chunks
    header_budget = min_chunk_budget
    n_header_chunks = n_chunks - len(topk_indices)
    header_total = header_budget * n_header_chunks

    # Remaining budget for top-k chunks
    remaining = max(total_budget - header_total, total_budget // 2)

    # Distribute to top-k based on their relative scores
    topk_list = list(topk_indices)
    topk_scores = np.array([scores_np[i] for i in topk_list])

    # Softmax among top-k only
    topk_exp = np.exp(topk_scores - topk_scores.max())
    topk_probs = topk_exp / topk_exp.sum()
    topk_budgets = (topk_probs * remaining).astype(int)

    # Ensure minimum
    topk_budgets = np.maximum(topk_budgets, min_chunk_budget * 2)

    # Build final budget list
    budgets = [header_budget] * n_chunks
    for i, idx in enumerate(topk_list):
        budgets[idx] = topk_budgets[i]

    return budgets


def build_compressed_context(
    raw_observation: str,
    chunks: List[str],
    budgets: List[int],
    compress_fn,
    total_budget: int,
    use_front_buffer: bool = True,
    use_anchors: bool = True,
) -> str:
    """
    Build final compressed context with front buffer and anchors.

    Args:
        raw_observation: Original observation string
        chunks: List of chunks
        budgets: Budget per chunk
        compress_fn: Function to compress a chunk to budget
        total_budget: Total budget target
        use_front_buffer: Whether to include front buffer
        use_anchors: Whether to include anchors

    Returns:
        Final compressed context string
    """
    parts = []

    # Add front buffer (35% of budget) for low budgets
    if use_front_buffer and total_budget <= LOW_BUDGET_THRESHOLD:
        front_budget = int(total_budget * FRONT_BUFFER_FRAC)
        front_buffer = raw_observation[:front_budget]
        # Try to end at word boundary
        last_space = front_buffer.rfind(' ')
        if last_space > front_budget * 0.7:
            front_buffer = front_buffer[:last_space]
        parts.append(f"[CONTEXT START]\n{front_buffer.strip()}")

    # Add anchors (15% of budget) for low budgets
    if use_anchors and total_budget <= LOW_BUDGET_THRESHOLD:
        anchor_budget = int(total_budget * ANCHOR_FRAC)
        anchors = extract_anchors(raw_observation, anchor_budget)
        if anchors:
            parts.append(f"[KEY INFO] {anchors}")

    # Add compressed chunks
    compressed_chunks = []
    for chunk, budget in zip(chunks, budgets):
        compressed = compress_fn(chunk, budget)
        if compressed and len(compressed.strip()) > 0:
            compressed_chunks.append(compressed.strip())

    if compressed_chunks:
        parts.append("\n".join(compressed_chunks))

    return "\n\n".join(parts)
