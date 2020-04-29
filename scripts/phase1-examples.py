import os
import json
import glob
import argparse
import multiprocessing

from tqdm import tqdm

from valerie.utils import get_logger
from valerie.data import Claim, Article
from valerie.processors import MultiClaimSupportProcessor

_logger = get_logger()


def load_claims(claims_file):
    _logger.info("... loading claims from %s ...", claims_file)
    with open(claims_file, 'r') as fi:
        claims = [Claim.from_dict(d) for d in tqdm(json.load(fi))]
    return claims


def load_articles(articles_dir, nproc):
    _logger.info("... loading articles from %s ...", articles_dir)
    articles = {}
    filepaths = glob.glob(os.path.join(articles_dir, "*.txt"))
    with multiprocessing.Pool(nproc) as pool:
        articles = {
            article_id: article
            for article_id, article in
            tqdm(pool.imap_unordered(_load_article, filepaths), total=len(filepaths))
        }
    return articles


def _load_article(filepath):
    with open(filepath) as fi:
        article_id = int(os.path.splitext(os.path.basename(filepath))[0])
        article = Article.from_txt(article_id, fi.read())
    return article_id, article


def load_processor(articles, word2vec_file, min_examples, max_examples, min_threshold, nproc):
    _logger.info("... loading processor ...")
    processor = MultiClaimSupportProcessor(
        articles=articles,
        word2vec_file=word2vec_file,
        min_examples=min_examples,
        max_examples=max_examples,
        min_threshold=min_threshold,
    )
    return processor


def generate_examples(claims, processor, nproc):
    _logger.info("... generating examples ...")

    _inputs = [[claim] for claim in claims]
    globals()["_proc"] = processor
    all_examples = []

    with multiprocessing.Pool(nproc) as pool:
        for examples in tqdm(pool.imap_unordered(_generate_examples, _inputs), total=len(_inputs)):
            all_examples.extend(examples)

    globals()["_proc"] = None
    return all_examples


def _generate_examples(claims):
    return _proc.generate_examples(claims) #pylint: disable=undefined-variable


def generate_train_test_examples(claims, processor, train_test_split_ratio, nproc):
    split_num = int(len(claims) * train_test_split_ratio)
    train_claims, test_claims = claims[:split_num], claims[split_num:]

    _logger.info("train_test_split_ratio = %f ...", train_test_split_ratio)
    _logger.info("      num_total_claims = %d ...", len(claims))
    _logger.info("   num_training_claims = %d ...", len(train_claims))
    _logger.info("    num_testing_claims = %d ...", len(test_claims))

    _logger.info("... generating train examples ...")
    train_examples = generate_examples(train_claims, processor, nproc)

    _logger.info("... generating test examples ...")
    test_examples = generate_examples(test_claims, processor, nproc)

    _logger.info("    num_total_examples = %d ...", len(train_examples+test_examples))
    _logger.info(" num_training_examples = %d ...", len(train_examples))
    _logger.info("  num_testing_examples = %d ...", len(test_examples))

    return train_examples, test_examples


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generate phase1 examples.")
    parser.add_argument("--claims_file", type=str)
    parser.add_argument("--articles_dir", type=str)
    parser.add_argument("--word2vec_file", type=str)
    parser.add_argument("--examples_file", type=str)
    parser.add_argument("--min_examples", type=int)
    parser.add_argument("--max_examples", type=int)
    parser.add_argument("--min_threshold", type=float)
    parser.add_argument("--nproc", type=int)
    parser.add_argument("--train_test_split_ratio", type=float, default=None)

    args = parser.parse_args()
    claims = load_claims(args.claims_file)
    articles = load_articles(args.articles_dir, args.nproc)
    processor = load_processor(
        articles,
        args.word2vec_file,
        args.min_examples,
        args.max_examples,
        args.min_threshold,
        args.nproc
    )

    if args.train_test_split_ratio is None:
        examples = generate_examples(claims, processor, args.nproc)
        _logger.info("... saving examples to %s ...", args.examples_file)
        with open(args.examples_file, 'w') as fo:
            json.dump([example.__dict__ for example in examples], fo)
    else:
        train_examples, test_examples = generate_train_test_examples(
            claims,
            processor,
            args.train_test_split_ratio,
            args.nproc
        )
        _logger.info("... saving examples to %s ...", args.examples_file)
        with open(args.examples_file, 'w') as fo:
            json.dump({
                "training_examples": [example.__dict__ for example in train_examples],
                "testing_examples": [example.__dict__ for example in test_examples],
            }, fo)


