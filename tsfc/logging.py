"""Logging for TSFC."""

import logging


ANSI_RED = "\x1b[1;31m"
ANSI_GREEN = "\x1b[1;32m"
ANSI_BLUE = "\x1b[1;34m"
ANSI_RESET = "\x1b[0m"


class ColourFormatter(logging.Formatter):
    """Apply colour to TSFC console output according to the log level."""

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record with the colour for its level.

        Parameters
        ----------
        record : logging.LogRecord
            The record to format.

        Returns
        -------
        str
            The formatted record.
        """
        message = super().format(record)
        if record.levelno >= logging.WARNING:
            return f"{ANSI_RED}{message}{ANSI_RESET}"
        if record.levelno == logging.INFO:
            return f"{ANSI_GREEN}{message}{ANSI_RESET}"
        return message


logger = logging.getLogger('tsfc')
_handler = logging.StreamHandler()
_handler.setFormatter(ColourFormatter())
logger.addHandler(_handler)
