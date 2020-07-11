"""Data classes and loading functions."""
import os
import glob
import json
import shutil
import logging
import collections
import multiprocessing

import bs4
import tldextract
from tqdm.auto import tqdm

from .preprocessing import clean_text

_logger = logging.getLogger(__name__)


class Claim:
    """A claim."""

    def __init__(
        self,
        id,
        claim,
        claimant=None,
        label=None,
        date=None,
        related_articles=None,
        explanation=None,
        support=None,
        dataset_name=None,
    ):
        """Constructor for Claim."""
        self.id = id
        self.claim = clean_text(claim)[:4000] if claim else None  # restrict num chars
        self.claimant = clean_text(claimant) if claimant else None
        self.label = label
        self.date = str(date)
        self.related_articles = related_articles
        self.explanation = explanation
        self.support = support
        self.dataset_name = dataset_name

        if dataset_name:
            self.index = dataset_name + "/" + str(self.id)
        else:
            self.index = self.id

    def to_dict(self):
        return self.__dict__

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

    def __repr__(self):
        return json.dumps(self.__dict__, indent=2)

    def __eq__(self, other):
        return self.claim == other.claim

    def __hash__(self):
        return hash(self.claim)


class Article:
    """An article."""

    def __init__(
        self,
        id,
        content=None,
        title=None,
        source=None,
        url=None,
        date=None,
        dataset_name=None,
    ):
        """Constructor for Article."""
        self.id = id
        self.title = title[:4000] if title else None  # restrict num chars
        self.content = (
            content[:16000] if content else None
        )  # restrict num chars in claim
        self.source = tldextract.extract(url).domain if url else None
        self.url = url
        self.date = date
        self.dataset_name = dataset_name

        if dataset_name:
            self.index = dataset_name + "/" + str(self.id)
        else:
            self.index = self.id

    def to_dict(self):
        return self.__dict__

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

    @classmethod
    def from_txt(cls, id, text, **kwargs):
        """Construct an Article given text."""
        title = text.partition("\n")[0]
        text = clean_text(text)
        return cls(id, content=text, title=title, **kwargs)

    @classmethod
    def from_html(cls, id, html, **kwargs):
        """Constructs an Article given an html text."""

        def tag_visible(element):
            whitelist = ["h1", "h2", "h3", "h4", "h5", "body", "p", "font"]
            if element.parent.name not in whitelist:
                return False
            if isinstance(element, bs4.Comment):
                return False
            return True

        soup = bs4.BeautifulSoup(html, "html.parser")

        # if not valid html, might be already preprocessed text
        if not bool(soup.find()):
            text = clean_text(html)
            return cls(id, content=text, **kwargs)

        texts = soup.findAll(text=True)
        texts = filter(tag_visible, texts)

        text = ""
        for t in texts:
            t = clean_text(t)
            if t and len(t) > 32:  # dissallow empty/short text sequences
                text += t + " "

        if "title" not in kwargs:
            title = soup.title if soup.title and soup.title.string else None
            title = clean_text(title.string) if title else None
        return cls(id, content=text, title=title, **kwargs)

    def __repr__(self):
        return json.dumps(self.__dict__, indent=2)

    def __eq__(self, other):
        return self.index == other.index

    def __hash__(self):
        return hash(self.index)


def combine_claims(claims_lists, logging_names=None):
    """The head of the list has most priority during union, the tail has the least."""
    if logging_names and len(claims_lists) != len(logging_names):
        raise ValueError("len claims_list must be equal to len logging_names")
    _logger.info("... combining claims ...")
    combined_claims_set = set()
    for i, claims in enumerate(claims_lists):
        prev_len = len(combined_claims_set)
        combined_claims_set = combined_claims_set | set(claims)
        _logger.info(
            "%s: %d --> %d (+ %d = %d - %d)",
            logging_names[i] if logging_names else str(i),
            prev_len,
            len(combined_claims_set),
            len(combined_claims_set) - prev_len,
            len(claims),
            prev_len + len(claims) - len(combined_claims_set),
        )
    return list(combined_claims_set)


def _save_relevant_articles_phase1(metadata, articles_dir, output_dir):
    # find set of relevant articles in trimmed down dataset
    relevant_articles = set()
    for d in metadata:
        relevant_articles.update(d["related_articles"])

    # copy relevant articles to output directory
    for fpath in tqdm(glob.glob(os.path.join(articles_dir, "*.txt"))):
        article_id = os.path.basename(fpath).split(".")[0]
        if int(article_id) in relevant_articles:
            shutil.copyfile(
                fpath, os.path.join(output_dir, "articles", os.path.basename(fpath))
            )

    return len(relevant_articles)


def _save_relevant_articles_phase2(metadata, articles_dir, output_dir):
    # find set of relevant articles in trimmed down dataset
    relevant_articles = set()
    for d in metadata:
        relevant_articles.update(
            int(os.path.basename(n).split(".")[0]) for n in d["related_articles"]
        )

    # copy relevant articles to output directory
    for fpath in tqdm(glob.glob(os.path.join(articles_dir, "*.html"))):
        article_id = os.path.basename(fpath).split(".")[0]
        if int(article_id) in relevant_articles:
            shutil.copyfile(
                fpath, os.path.join(output_dir, "articles", os.path.basename(fpath))
            )

    return len(relevant_articles)


def trim_metadata_phase1(metadata_file, articles_dir, output_dir, n_examples):
    os.makedirs(os.path.join(output_dir, "articles"))

    # load data
    with open(metadata_file, "r") as fi:
        raw_data = json.load(fi)

    # trim down dataset
    _logger.info("orig len: %d", len(raw_data))
    metadata = raw_data[:n_examples]
    _logger.info("new len: %d", len(metadata))

    num_articles = _save_relevant_articles_phase1(metadata, articles_dir, output_dir)
    _logger.info("len relevant articles set: %d", num_articles)

    # save trimmed down metadata.json to output directory
    with open(os.path.join(output_dir, "metadata.json"), "w") as fo:
        json.dump(metadata, fo, indent=2)


def trim_metadata_phase2(metadata_file, articles_dir, output_dir, n_examples=None):
    os.makedirs(os.path.join(output_dir, "articles"))

    # load data
    with open(metadata_file, "r") as fi:
        raw_data = json.load(fi)

    # trim down dataset
    _logger.info("orig len: %d", len(raw_data))
    metadata = raw_data[:n_examples]
    _logger.info("new len: %d", len(metadata))

    num_articles = _save_relevant_articles_phase2(metadata, articles_dir, output_dir)
    _logger.info("len relevant articles set: %d", num_articles)

    # save trimmed down metadata.json to output directory
    with open(os.path.join(output_dir, "metadata.json"), "w") as fo:
        json.dump(metadata, fo, indent=2)


def train_test_split_phase2(
    metadata_file, articles_dir, train_dir, test_dir, train_size, random_state
):
    from sklearn.model_selection import train_test_split

    if os.path.exists(train_dir):
        raise ValueError("train_dir ({}) already exists".format(train_dir))
    if os.path.exists(test_dir):
        raise ValueError("test_dir ({}) already exists".format(test_dir))
    os.makedirs(os.path.join(train_dir, "articles"))
    os.makedirs(os.path.join(test_dir, "articles"))

    with open(metadata_file, "r") as fi:
        metadata = json.load(fi)

    # log args
    _logger.info("metadata_file:  %s", metadata_file)
    _logger.info("articles_dir: %s", articles_dir)
    _logger.info("train_dir:    %s", train_dir)
    _logger.info("test_dir:     %s", test_dir)
    _logger.info("train_size:   %.2f", train_size)
    _logger.info("random_state: %d", random_state)
    _logger.info("")

    # train_test_split
    all_labels = [d["label"] for d in metadata]
    training_data, testing_data, _, _ = train_test_split(
        metadata,
        all_labels,
        stratify=all_labels,
        train_size=train_size,
        random_state=random_state,
    )

    # logging train test split
    _logger.info("Num Total Claims:     %d", len(metadata))
    _logger.info("Num Train Claims:     %d", len(training_data))
    _logger.info("Num Test Claims:      %d", len(testing_data))
    _logger.info("")
    all_labels_count = collections.Counter(all_labels)
    train_labels_count = collections.Counter([d["label"] for d in training_data])
    test_labels_count = collections.Counter([d["label"] for d in testing_data])
    _logger.info("All Labels Count:     %s", str(dict(all_labels_count)))
    _logger.info("Train Labels Count:   %s", str(dict(train_labels_count)))
    _logger.info("Test Labels Count:    %s", str(dict(test_labels_count)))
    _logger.info("")

    # articles
    num_total_articles = len(glob.glob(os.path.join(articles_dir, "*.html")))
    _logger.info("Num Total Articles:   %d", num_total_articles)
    num_train_articles = _save_relevant_articles_phase2(
        training_data, articles_dir, train_dir
    )
    _logger.info("Num Train Articles:   %d", num_train_articles)
    num_test_articles = _save_relevant_articles_phase2(
        testing_data, articles_dir, test_dir
    )
    _logger.info("Num Test Articles:    %d", num_test_articles)

    # write metadata
    with open(os.path.join(train_dir, "metadata.json"), "w") as fo:
        json.dump(training_data, fo, indent=2)
    with open(os.path.join(test_dir, "metadata.json"), "w") as fo:
        json.dump(testing_data, fo, indent=2)
