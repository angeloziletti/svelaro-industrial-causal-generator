"""
Logger configuration for the generator.
This module provides a centralized logging setup for the project.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


# Global flag to track if main logger has been initialized
_MAIN_LOGGER_INITIALIZED = False

# Global variable to track current log file path
_CURRENT_LOG_FILE_PATH = None






def setup_logger(
    log_level="INFO",  # Accept string or int
    calculations_dir=None,
    log_folder="logs",
    log_to_console=True,
    log_to_file=True,
    log_file_prefix="extrusion_defect_generator",
    log_file_path=None,
):
    """
    Set up a logger for the extrusion defect generator.

    IMPORTANT: This should only be called ONCE at the start of your application!
    Use get_logger() for subsequent logger access.

    Args:
        log_level: Logging level as string or int constant
        calculations_dir: Directory for logs (if None, uses ./logs)
        log_folder: Subfolder name for logs
        log_to_console: Whether to log to console
        log_to_file: Whether to log to file
        log_file_prefix: Prefix for log file names
        log_file_path: Optional explicit path (str or Path). When set, skips
            calculations_dir/log_folder/log_file_prefix/timestamp generation
            and writes directly to this path in append mode. Used by worker
            processes that need to share a single log file with the master.
    """
    global _MAIN_LOGGER_INITIALIZED, _CURRENT_LOG_FILE_PATH

    # Convert string log level to constant if needed
    if isinstance(log_level, str):
        log_level = get_log_level(log_level)

    # Create the main logger
    logger = logging.getLogger("extrusion_defect_generator")

    # CRITICAL FIX: Only set up if not already initialized
    if _MAIN_LOGGER_INITIALIZED and logger.handlers:
        # Logger already exists - check if we need to add file logging
        if log_to_file and _CURRENT_LOG_FILE_PATH is None:
            # File logging requested but not currently enabled - add it
            if calculations_dir is None:
                log_dir = Path("./logs")
            else:
                log_dir = Path(calculations_dir) / log_folder

            log_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = log_dir / f"{log_file_prefix}_{timestamp}.log"
            _CURRENT_LOG_FILE_PATH = log_file

            detailed_formatter = logging.Formatter(
                "%(asctime)s - %(filename)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(detailed_formatter)
            logger.addHandler(file_handler)
            logger.info(f"Logs will be saved to: {log_file}")

        return logger

    # Set logger level
    logger.setLevel(log_level)

    # CRITICAL FIX: Remove any existing handlers to prevent duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create formatters
    simple_formatter = logging.Formatter(
        "%(asctime)s - %(filename)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    detailed_formatter = logging.Formatter(
        "%(asctime)s - %(filename)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Add console handler if requested
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(simple_formatter)
        logger.addHandler(console_handler)

    # Add file handler if requested
    if log_to_file:
        # Explicit path wins (worker processes attaching to the master log);
        # otherwise build a fresh timestamped path under calculations_dir.
        if log_file_path is not None:
            log_file = Path(log_file_path)
            log_file.parent.mkdir(parents=True, exist_ok=True)
        else:
            if calculations_dir is None:
                log_dir = Path("./logs")
            else:
                log_dir = Path(calculations_dir) / log_folder
            log_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = log_dir / f"{log_file_prefix}_{timestamp}.log"

        _CURRENT_LOG_FILE_PATH = log_file

        # FileHandler defaults to mode='a' (append) — multiple processes can
        # all write to the same file. POSIX guarantees atomic appends below
        # PIPE_BUF (4KB); Windows is best-effort but works for typical log
        # lines. Cross-worker interleaving is benign because each worker
        # prefixes its messages with [g=N r=M].
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)

        logger.info(f"Logs will be saved to: {log_file}")

    # Mark as initialized
    _MAIN_LOGGER_INITIALIZED = True

    # Prevent propagation to root logger to avoid duplicate messages
    logger.propagate = False

    return logger


def get_logger(name=None):
    """
    Get the configured logger without creating duplicates.

    This is the function you should use throughout your code!
    It will NOT create duplicate loggers or log messages.

    Args:
        name (str, optional): If provided, creates a child logger. If None, returns main logger.

    Returns:
        logging.Logger: The configured logger instance
    """
    global _MAIN_LOGGER_INITIALIZED

    main_logger_name = "extrusion_defect_generator"

    # Get the main logger
    main_logger = logging.getLogger(main_logger_name)

    # CRITICAL FIX: Only initialize if never initialized before
    if not _MAIN_LOGGER_INITIALIZED or not main_logger.handlers:
        # Initialize with minimal setup (no file logging by default in get_logger)
        main_logger = setup_logger(
            log_level="INFO",
            log_to_file=False,  # Don't create file handlers in get_logger calls
            log_to_console=True,
        )

    # Return appropriate logger
    if name is None:
        return main_logger
    else:
        # Return child logger (inherits from main)
        child_logger = logging.getLogger(f"{main_logger_name}.{name}")
        # Child loggers inherit handlers from parent, so no need to add handlers
        return child_logger










def get_log_level(level_name):
    """
    Convert a string log level name to the corresponding logging constant.

    Args:
        level_name: String name of log level ('DEBUG', 'INFO', etc.)

    Returns:
        The corresponding logging level constant
    """
    log_level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    return log_level_map.get(level_name, logging.INFO)


# OPTIONAL: Auto-configure basic logging on import (if not already configured)


# Uncomment this if you want automatic basic setup on import
# (But it's better to explicitly call setup_logger() once at the start)
# _auto_configure_if_needed()
