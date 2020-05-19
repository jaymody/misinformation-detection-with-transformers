import os
import glob
import json
import shutil
import argparse

from tqdm import tqdm

from valerie.utils import get_logger

_logger = get_logger()


def trim_dataset(claims_file, articles_dir, output_dir, n_examples, nproc=1):
    # if output_dir already exists, raise error, else make necessary dirs
    if os.path.exists(output_dir):
        raise ValueError("output_dir ({}) already exists".format(output_dir))
    os.makedirs(os.path.join(output_dir, "articles"))

    # load data
    with open(claims_file, "r") as fi:
        raw_data = json.load(fi)

    # trim down dataset
    _logger.info("orig len: %d", len(raw_data))
    data = raw_data[:n_examples]
    _logger.info("new len: %d", len(raw_data))

    # find set of relevant articles in trimmed down dataset
    relevant_articles = set()
    for d in data:
        relevant_articles.update(
            int(os.path.basename(n).split(".")[0]) for n in d["related_articles"]
        )
    _logger.info("len relevant articles set: %d", len(relevant_articles))

    # copy relevant articles to output directory
    for fpath in tqdm(glob.glob(os.path.join(articles_dir, "*.html"))):
        article_id = os.path.basename(fpath).split(".")[0]
        if int(article_id) in relevant_articles:
            shutil.copyfile(
                fpath, os.path.join(output_dir, "articles", os.path.basename(fpath))
            )

    # save trimmed down metadata.json to output directory
    with open(os.path.join(output_dir, "metadata.json"), "w") as fo:
        json.dump(data, fo, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Trims down the leader's prize phase2 dataset (for quicker testing)."
    )
    parser.add_argument("--claims_file", type=str)
    parser.add_argument("--articles_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--n_examples", type=int)
    parser.add_argument("--nproc", type=int)

    kwargs = parser.parse_args()
    trim_dataset(**kwargs.__dict__)
