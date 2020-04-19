"""Utility functions."""
import logging

_logger = logging.getLogger(__name__)


def get_logger(logfile=None):
    """Gets a nicely formatted logger."""
    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s:%(name)s: %(message)s')
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers = [sh]

    if logfile:
        fh = logging.FileHandler(logfile)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.handlers.append(fh)

    return logger
