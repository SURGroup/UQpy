import logging

"""Based on https://stackoverflow.com/questions/14844970/
modifying-logging-message-format-based-on-message-logging-level-in-python3"""


class UQpyLoggingFormatter(logging.Formatter):
    def __init__(self):
        super().__init__(fmt="%(levelno)d: %(msg)s", datefmt=None, style="%")
        self.format_dictionary = {
            logging.NOTSET: "%(message)s",
            logging.DEBUG: "%(message)s",
            logging.INFO: "[%(levelname)s] - %(asctime)s - %(message)s",
            logging.WARNING: "[%(levelname)s] - %(asctime)s - File: %(filename)s - %(message)s",
            logging.WARN: "[%(levelname)s] - %(asctime)s - File: %(filename)s - %(message)s",
            logging.ERROR: "[%(levelname)s] - %(asctime)s - File: %(filename)s - Method: %(funcName)s -"
            " Line: %(lineno)s - %(message)s",
            logging.CRITICAL: "[%(levelname)s] - %(asctime)s - File: %(filename)s - Method: %(funcName)s -"
            " Line: %(lineno)s - %(message)s",
        }

    def format(self, record):

        # Save the original format configured by the user
        # when the logger formatter was instantiated
        format_orig = self._style._fmt

        # Replace the original format with one customized by logging level
        self._style._fmt = self.format_dictionary[record.levelno]

        # Call the original formatter class to do the grunt work
        result = logging.Formatter.format(self, record)

        # Restore the original format configured by the user
        self._style._fmt = format_orig

        return result
