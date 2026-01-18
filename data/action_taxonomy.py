"""Coarse action classification for tractable next-action prediction."""

from typing import Dict, Tuple
import re


# ============================================================================
# WebShop Action Taxonomy
# ============================================================================

WEBSHOP_ACTIONS = {
    'search': 0,
    'click_product': 1,
    'click_option': 2,
    'click_buy': 3,
    'click_back': 4,
    'click_nav': 5,
    'unknown': 6,
}

WEBSHOP_ACTION_NAMES = {v: k for k, v in WEBSHOP_ACTIONS.items()}


def classify_webshop_action(action_str: str) -> Tuple[int, str]:
    """
    Map raw WebShop action string to coarse action class.

    Returns:
        (action_id, action_name)
    """
    action_str_lower = action_str.lower().strip()

    # Search actions
    if action_str_lower.startswith('search['):
        return WEBSHOP_ACTIONS['search'], 'search'

    # Must be a click action
    if not action_str_lower.startswith('click['):
        return WEBSHOP_ACTIONS['unknown'], 'unknown'

    # Extract click target
    match = re.match(r'click\[(.+)\]', action_str, re.IGNORECASE)
    if not match:
        return WEBSHOP_ACTIONS['unknown'], 'unknown'

    target = match.group(1).lower()

    # Buy/cart actions
    buy_keywords = ['buy now', 'add to cart', 'purchase', 'checkout']
    if any(kw in target for kw in buy_keywords):
        return WEBSHOP_ACTIONS['click_buy'], 'click_buy'

    # Back/navigation
    back_keywords = ['back', 'return', '< prev', 'previous']
    if any(kw in target for kw in back_keywords):
        return WEBSHOP_ACTIONS['click_back'], 'click_back'

    # Pagination
    nav_keywords = ['next', 'page', '>', 'more results']
    if any(kw in target for kw in nav_keywords):
        return WEBSHOP_ACTIONS['click_nav'], 'click_nav'

    # Options are typically short (colors, sizes)
    # Products are longer (full product names)
    if len(target) < 25:
        # Check if it looks like an option (color, size, etc.)
        option_patterns = [
            r'^(small|medium|large|xl|xxl|xs)$',
            r'^(red|blue|green|black|white|pink|purple|yellow|orange|brown|gray|grey)$',
            r'^\d+(\.\d+)?\s*(oz|ml|gb|tb|inch|in|cm|mm)$',
            r'^size\s*\d+',
            r'^\d+\s*pack$',
        ]
        for pattern in option_patterns:
            if re.match(pattern, target):
                return WEBSHOP_ACTIONS['click_option'], 'click_option'

        # Short but not obviously an option - probably still an option
        return WEBSHOP_ACTIONS['click_option'], 'click_option'

    # Longer text = product name
    return WEBSHOP_ACTIONS['click_product'], 'click_product'


# ============================================================================
# ALFWorld Action Taxonomy
# ============================================================================

ALFWORLD_ACTIONS = {
    'go': 0,
    'open': 1,
    'close': 2,
    'take': 3,
    'put': 4,
    'toggle': 5,
    'heat': 6,
    'cool': 7,
    'clean': 8,
    'examine': 9,
    'look': 10,
    'inventory': 11,
    'use': 12,
    'unknown': 13,
}

ALFWORLD_ACTION_NAMES = {v: k for k, v in ALFWORLD_ACTIONS.items()}


def classify_alfworld_action(action_str: str) -> Tuple[int, str]:
    """
    Map raw ALFWorld action string to coarse action class.

    Returns:
        (action_id, action_name)
    """
    action_str_lower = action_str.lower().strip()

    # Match action prefix
    for action_name, action_id in ALFWORLD_ACTIONS.items():
        if action_name == 'unknown':
            continue
        if action_str_lower.startswith(action_name):
            return action_id, action_name

    return ALFWORLD_ACTIONS['unknown'], 'unknown'


# ============================================================================
# Unified Interface
# ============================================================================

def classify_action(action_str: str, env: str) -> Tuple[int, str]:
    """Classify action for any supported environment."""
    if env == 'webshop':
        return classify_webshop_action(action_str)
    elif env == 'alfworld':
        return classify_alfworld_action(action_str)
    else:
        raise ValueError(f"Unknown environment: {env}")


def get_num_actions(env: str) -> int:
    """Get number of action classes for an environment."""
    if env == 'webshop':
        return len(WEBSHOP_ACTIONS)
    elif env == 'alfworld':
        return len(ALFWORLD_ACTIONS)
    else:
        raise ValueError(f"Unknown environment: {env}")


def get_action_name(action_id: int, env: str) -> str:
    """Get action name from ID."""
    if env == 'webshop':
        return WEBSHOP_ACTION_NAMES.get(action_id, 'unknown')
    elif env == 'alfworld':
        return ALFWORLD_ACTION_NAMES.get(action_id, 'unknown')
    else:
        raise ValueError(f"Unknown environment: {env}")
