"""
Structured logging module for the Oil Spill Detection System.
Provides formatted loggers with timestamps, scene IDs, module names, and severity levels.
"""
import logging
import sys
from typing import Optional


def get_logger(name: str, scene_id: Optional[str] = None) -> logging.Logger:
    """
    Get or configure a structured logger.
    
    Args:
        name: Name of the module/logger.
        scene_id: Optional satellite scene ID context.
        
    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        
        prefix = f"[{scene_id}] " if scene_id else ""
        formatter = logging.Formatter(
            f"%(asctime)s | %(levelname)-7s | %(name)s | {prefix}%(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
        
    return logger
