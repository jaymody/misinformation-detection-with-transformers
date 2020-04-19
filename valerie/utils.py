"""Utility functions."""
import logging

import gensim

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


def load_word2vec(word2vec_path):
    _logger.info("... loading word2vec model ...")
    return gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
