"""Logging helpers.

Library code uses ``get_logger(__name__).info(...)`` and never configures
handlers; applications / CLI scripts call ``configure_logging()`` once so the
messages surface (default format is message-only, so output reads like prints).
"""

from __future__ import annotations

import logging


def get_logger(name: str = "vlm_medseg") -> logging.Logger:
    return logging.getLogger(name)


def configure_logging(level: int = logging.INFO, fmt: str = "%(message)s") -> None:
    logging.basicConfig(level=level, format=fmt)
    logging.getLogger("vlm_medseg").setLevel(level)
