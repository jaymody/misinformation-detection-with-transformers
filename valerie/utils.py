"""Utility functions."""
import os
import json
import logging

from tqdm import tqdm

_logger = logging.getLogger(__name__)


def get_logger(logfile=None):
    """Gets a nicely formatted logger."""
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s:%(name)s: %(message)s")
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


def download_embedding_model(model_name, save_dir):
    # see https://github.com/RaRe-Technologies/gensim-data for more details
    import gensim.downloader as api

    _logger.info("... downloading {} ...".format(model_name))
    src = api.load(model_name, return_path=True)
    dst = os.path.join(save_dir, os.path.basename(src))
    os.rename(src, dst)
    _logger.info("... moved from {} to {}...".format(src, dst))


def load_word2vec(word2vec_path):
    from gensim.models import KeyedVectors

    _logger.info("... loading word2vec model ...")
    return KeyedVectors.load_word2vec_format(word2vec_path, binary=True)


def load_claims(filepath, as_list=False, verbose=True):
    from valerie.data import Claim

    with open(filepath) as fi:
        claims = {
            k: Claim.from_dict(v)
            for k, v in tqdm(
                json.load(fi).items(), desc="loading claims", disable=not verbose
            )
        }

    if as_list:
        return sorted(list(claims.values()), key=lambda x: x.id)
    return claims


def save_claims(filepath, claims, **kwargs):
    with open(filepath, "w") as fo:
        json.dump({k: v.to_dict() for k, v in claims.items()}, fo, **kwargs)


def load_articles(filepath, as_list=False, verbose=True):
    from valerie.data import Article

    with open(filepath) as fi:
        articles = {
            k: Article.from_dict(v)
            for k, v in tqdm(
                json.load(fi).items(), desc="loading articles", disable=not verbose,
            )
        }

    if as_list:
        return sorted(list(articles.values()), key=lambda x: x.id)
    return articles


def save_articles(filepath, articles, **kwargs):
    with open(filepath, "w") as fo:
        json.dump({k: v.to_dict() for k, v in articles.items()}, fo, **kwargs)
