from __future__ import annotations

import logging


def get_logger(log_level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("paku")
    if logger.handlers:
        return logger

    level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(level)

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
    logger.addHandler(handler)

    return logger
