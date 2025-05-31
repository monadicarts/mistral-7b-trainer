import logging
import sys


class AnsiColorFormatter(logging.Formatter):
    """
    Custom logging formatter for colored terminal output.
    Timestamp, log level, and message are colored separately.
    """

    TIMESTAMP_COLOR = "\033[35m"  # Magenta
    LEVEL_COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[41m",  # Red background
    }
    RESET = "\033[0m"

    def format(self, record):
        # Format timestamp
        asctime = self.formatTime(record, self.datefmt)
        timestamp = f"{self.TIMESTAMP_COLOR}{asctime}{self.RESET}"

        # Format log level
        level_color = self.LEVEL_COLORS.get(record.levelname, self.RESET)
        levelname = f"{level_color}{record.levelname}{self.RESET}"

        # Message (default color)
        message = record.getMessage()

        # Compose final log line
        return f"[{timestamp}] {levelname}: {message}"


def get_logger(name: str = __name__, level: int = logging.INFO) -> logging.Logger:
    """
    Returns a logger instance with colored output.

    Args:
        name (str): Logger name.
        level (int): Logging level.

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(AnsiColorFormatter())
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger
