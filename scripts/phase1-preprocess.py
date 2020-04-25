import os
import json
import glob
import argparse
import multiprocessing

from tqdm import tqdm

from valerie.utils import get_logger
from valerie.data import Claim, Article
from valerie.modeling import InputExample
from valerie.preprocessing import ClaimPreprocessor

_logger = get_logger()


def visit(claim):
    support = preprocessor.generate_support(claim, keep_n, min_threshold, min_examples) #pylint: disable=undefined-variable
    examples = []
    for s in support:
        examples.append(InputExample(
            guid=claim.id,
            text_a=claim.claim,
            text_b=s["text"],
            label=claim.label
        ))
    return examples


def generate_examples(claims_file, articles_dir, examples_file, word2vec_file, keep_n, min_threshold, min_examples, nproc):
    _logger.info("... loading claims from {} ...".format(claims_file))
    with open(claims_file, 'r') as fi:
        claims = [Claim.from_dict(d) for d in tqdm(json.load(fi))]

    _logger.info("... loading articles from {} ...".format(articles_dir))
    articles = {}
    for filepath in tqdm(glob.glob(os.path.join(articles_dir, "*.txt"))):
        with open(filepath) as fi:
            article_id = int(os.path.splitext(os.path.basename(filepath))[0])
            articles[article_id] = Article.from_txt(article_id, fi.read())

    _logger.info("... initializing preprocessor ...")
    preprocessor = ClaimPreprocessor(articles, word2vec_file)
    globals()["preprocessor"] = preprocessor
    globals()["keep_n"] = keep_n
    globals()["min_threshold"] = min_threshold
    globals()["min_examples"] = min_examples

    _logger.info("... generating examples ...")
    all_examples = []
    pool = multiprocessing.Pool(nproc)
    for examples in tqdm(pool.imap_unordered(visit, claims), total=len(claims)):
        all_examples.extend(examples)

    _logger.info("... saving examples to {} ...".format(examples_file))
    with open(examples_file, 'w') as fo:
        json.dump([example.__dict__ for example in all_examples], fo)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Phase1 preprocess data.")
    parser.add_argument("--claims_file", type=str)
    parser.add_argument("--articles_dir", type=str)
    parser.add_argument("--examples_file", type=str)
    parser.add_argument("--word2vec_file", type=str)
    parser.add_argument("--keep_n", type=int)
    parser.add_argument("--min_threshold", type=float)
    parser.add_argument("--min_examples", type=int)
    parser.add_argument("--nproc", type=int)

    kwargs = parser.parse_args()
    generate_examples(**kwargs.__dict__)

