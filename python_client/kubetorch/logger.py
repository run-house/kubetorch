import logging
import os
import sys


def get_logger(name) -> logging.Logger:
    """
    Creates and returns a logger with the specified name.

    Ensures a universal logger configuration across the codebase with the format:
    "levelname - asctime - filename:lineno - message"

    Args:
        name (str): Name of the logger. Defaults to None, which gets the root logger.

    Returns:
        logging.Logger: Configured logger instance.
    """
    # Create or retrieve the logger
    return logging.getLogger(name)


class NewLineFormatter(logging.Formatter):
    """Adds logging prefix to newlines to align multi-line messages."""

    def __init__(self, fmt, datefmt=None):
        logging.Formatter.__init__(self, fmt, datefmt)

    def format(self, record):
        msg = logging.Formatter.format(self, record)
        if record.message != "":
            parts = msg.partition(record.message)
            msg = msg.replace("\n", "\r\n" + parts[0])
        return msg


root_logger = logging.getLogger("kubetorch")


def init_logger(logger):
    level = os.getenv("KT_LOG_LEVEL") or "INFO"
    level = getattr(logging, level.upper())
    logger.setLevel(level)
    for handler in logger.handlers:
        logger.removeHandler(handler)

    if not logger.handlers:
        formatter = NewLineFormatter(
            "%(levelname)s | %(asctime)s | %(name)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.propagate = False


init_logger(root_logger)
