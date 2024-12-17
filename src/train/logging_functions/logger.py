"""Module to provide logging functionality"""

import logging
from pathlib import Path


def setup_logging(log_file_path: Path, log_level: int = logging.DEBUG):
    """
    Sets up a logging system that logs messages to both the console and a log file.

    Args:
        log_file_path (Path): Path to the log file.
        log_level (int): Logging level (e.g., logging.INFO, logging. DEBUG).
    """
    # Create the log file directory if it doesn't exist
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    # Configure the logger
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Create formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logging.info("Logging system initialized.")
