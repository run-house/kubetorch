import logging
import logging.config
import os
from typing import Optional


try:
    from utils import ensure_structured_logging, LOG_CONFIG
except ImportError:
    from .utils import ensure_structured_logging, LOG_CONFIG


HEARTBEAT_INTERVAL_DIVISOR = 5

# Set up our structured JSON logging
logging.config.dictConfig(LOG_CONFIG)
ensure_structured_logging()

logger = logging.getLogger(__name__)
# Set log level based on environment variable
log_level = os.getenv("KT_LOG_LEVEL")
if log_level:
    log_level = log_level.upper()
    logger.setLevel(getattr(logging, log_level, logging.INFO))


def get_inactivity_ttl_annotation() -> Optional[int]:
    """
    Get the inactivity TTL from pod annotations.
    Returns TTL in seconds, or None if not found.
    """
    try:
        # Try to get from environment variable first (can be injected via downward API)
        ttl_str = os.getenv("KT_INACTIVITY_TTL")
        if ttl_str:
            return parse_ttl_string(ttl_str)
        return None

    except Exception as e:
        logger.error(f"Error getting pod TTL annotation: {e}")

    return None


def parse_ttl_string(ttl_str: str) -> Optional[int]:
    """Parse TTL string to seconds. Supports formats: 300, 5m, 1h, 1h30m, 1d"""
    ttl_str = ttl_str.strip().lower()

    # If it's just a number, assume seconds
    if ttl_str.isdigit():
        return int(ttl_str)

    # Parse duration strings
    total_seconds = 0
    import re

    # Match patterns like 1h, 30m, 45s
    pattern = r"(\d+)([dhms])"
    matches = re.findall(pattern, ttl_str)

    for value, unit in matches:
        value = int(value)
        if unit == "d":
            total_seconds += value * 24 * 3600
        elif unit == "h":
            total_seconds += value * 3600
        elif unit == "m":
            total_seconds += value * 60
        elif unit == "s":
            total_seconds += value

    return total_seconds if total_seconds > 0 else None
