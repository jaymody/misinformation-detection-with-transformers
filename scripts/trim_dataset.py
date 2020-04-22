import os
import copy
import glob
import json
import shutil
import argparse
import multiprocessing

import numpy as np
from tqdm import tqdm

from valerie.utils import get_logger

_logger = get_logger()


def trim_dataset(claims_file, articles_dir, output_dir, n_examples, nproc=1):
    # load data
    with open(claims_file, 'r') as fi:
        raw_data = json.load(fi)

    # trim down dataset
    _logger.info("orig len: %d", len(raw_data))
    claims = raw_data[:n_examples]
    _logger.info("new len: %d", len(claims))

    # find relevant articles to trimmed down dataset
    html_fpaths = glob.glob(os.path.join(articles_dir, "*.html"))
    txt_fpaths = glob.glob(os.path.join(articles_dir, "*.txt"))
    relevant_articles = set()

    if html_fpaths and txt_fpaths:
        raise ValueError("both html and txt files found in articles dir")
    elif html_fpaths:
        fpaths = html_fpaths
        for c in claims:
            relevant_articles.update(os.path.basename(n) for n in c["related_articles"])
    else:
        fpaths = txt_fpaths
        for c in claims:
            relevant_articles.update(str(a) + ".txt" for a in c["related_articles"])

    # create dirs
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if os.path.exists(os.path.join(output_dir, "articles")):
        raise ValueError("articles dir already exists in output dir")
    os.mkdir(os.path.join(output_dir, "articles"))

    # load relevant articles data
    _logger.info("len relevant articles set: %d", len(relevant_articles))
    for fpath in tqdm(fpaths):
        article = os.path.basename(fpath)
        if article in relevant_articles:
            shutil.copyfile(fpath, os.path.join(output_dir, "articles", article))

    #
    with open(os.path.join(output_dir, "metadata.json"), 'w') as fo:
        json.dump(claims, fo, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Trims down the leader's prize datasets (for quicker testing).")
    parser.add_argument("--claims_file", type=str)
    parser.add_argument("--articles_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--n_examples", type=int)
    parser.add_argument("--nproc", type=int)

    kwargs = parser.parse_args()
    trim_dataset(**kwargs.__dict__)
