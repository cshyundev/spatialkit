"""
Module Name: logger.py

Description:
This module initializes a logger with rich formatting using the Rich library and provides functions 
to add a file handler for logging messages to a file.
It supports logging at various levels, including INFO, DEBUG, WARNING, ERROR, and CRITICAL.

Main Functions:
- init_logger: Initializes and returns a logger with rich formatting.
- add_file_handler: Adds a file handler to the logger for logging to a specified file.

Author: Sehyun Cha
Email: cshyundev@gmail.com
Version: 0.2.1

License: MIT LICENSE
"""

import logging
from absl import logging as absl_logging
from rich.logging import RichHandler

RICH_FORMAT = "[%(filename)s:%(lineno)s] >> %(message)s"
FILE_HANDLER_FORMAT = (
    "[%(asctime)s] %(levelname)s [%(filename)s:%(funcName)s:%(lineno)s] >> %(message)s"
)


def init_logger() -> "logging":
    """
    Initializes a logger with rich formatting.

    Configures the logger to use the RichHandler for console output with a specified
    format and sets the logging level to INFO.

    Returns:
        logging: A configured logger instance.

    Example:
        logger = init_logger()
    """
    absl_logging.get_absl_handler().setLevel(logging.FATAL)
    logging.basicConfig(
        level="INFO",
        format=RICH_FORMAT,
        handlers=[RichHandler(rich_tracebacks=True)],
    )
    rich_logger = logging.getLogger("rich")
    return rich_logger


def add_file_handler(log_path: str):
    """
    Adds a file handler to the logger.

    Args:
        log_path (str): The path to the log file where messages will be written.

    Example:
        add_file_handler('app.log')
    """
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(FILE_HANDLER_FORMAT))
    logger.addHandler(file_handler)


logger = init_logger()

LOG_INFO = logger.info
LOG_DEBUG = logger.debug
LOG_WARN = logger.warning
LOG_ERROR = logger.error
LOG_CRITICAL = logger.critical

if __name__ == "__main__":
    LOG_INFO("information test")
    LOG_DEBUG("debug test")
    LOG_WARN("warn test")
    LOG_ERROR("error test")
    LOG_CRITICAL("critical test")
