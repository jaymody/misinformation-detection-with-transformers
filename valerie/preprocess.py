"""Text preprocessing."""
import os
import copy
import glob
import json
import logging
import unicodedata
import multiprocessing

import nltk
from tqdm import tqdm

_logger = logging.getLogger(__name__)


def _clean_text(text):
    output = []
    for char in text:
        cp = ord(char)
        if cp == 0 or cp == 0xfffd or _is_control(char):
            continue
        if _is_whitespace(char):
            output.append(" ")
        else:
            output.append(char)
    return "".join(output).strip()


def _is_whitespace(char):
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char):
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat in ("Cc", "Cf"):
        return True
    return False


def _split_sentences(text):
    return [sentence.strip() for sentence in nltk.tokenize.sent_tokenize(text)]


def _split_and_clean(x):
    return x[0], _split_sentences(_clean_text(x[1]))


def preprocess(data_path, articles_dir, nproc=1):
    # Load data
    _logger.info("... loading train data ...")
    with open(data_path, 'r', encoding="utf-8") as fi:
        raw_data = json.load(fi)

    _logger.info("... loading articles data ...")
    raw_articles_data = {}
    for fpath in tqdm(glob.glob(os.path.join(articles_dir, "*.txt"))):
        with open(fpath, 'r', encoding="utf-8") as fi:
            raw_articles_data[os.path.basename(fpath).split(".")[0]] = fi.read()

    # Clean claim text in training data
    _logger.info("... cleaning train data ...")
    data = copy.deepcopy(raw_data)
    for example in tqdm(data):
        example["claim"] = _clean_text(example["claim"])

    # Split articles into sentences and clean text
    _logger.info("... cleaning articles data ...")

    articles_data = {}
    pool = multiprocessing.Pool(nproc)

    for k,v in tqdm(pool.imap_unordered(_split_and_clean, raw_articles_data.items()), total=len(raw_articles_data)):
        articles_data[k] = v

    return data, articles_data
