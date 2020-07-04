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
    ):
        """Constructor for Claim."""
        self.id = id
        self.claim = clean_text(claim)[:4000] if claim else None  # restrict num chars
        self.claimant = clean_text(claimant) if claimant else None
        self.label = label
        self.date = date
        self.related_articles = related_articles
        self.explanation = explanation
        self.support = support

    def to_dict(self):
        return self.__dict__

    @classmethod
    def from_dict(cls, d):
        if "id" in d:
            _id = d.pop("id")
            return cls(_id, **d)
        return cls(**d)

    def __repr__(self):
        return json.dumps(self.__dict__, indent=2)

    def __eq__(self, other):
        return self.id == other.id


class Article:
    """An article."""

    def __init__(
        self,
        id,
        content=None,
        title=None,
        source=None,
        author=None,
        url=None,
        date=None,
    ):
        """Constructor for Article."""
        self.id = id
        self.title = title[:4000] if title else None  # restrict num chars
        self.content = (
            content[:16000] if content else None
        )  # restrict num chars in claim
        self.source = tldextract.extract(url).domain if url else None
        self.author = author
        self.url = url
        self.date = date

    def to_dict(self):
        return self.__dict__

    @classmethod
    def from_dict(cls, d):
        if "id" in d:
            _id = d.pop("id")
            return cls(_id, **d)
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
        return self.id == other.id


def load_claims(filepath, as_list=False, verbose=True):
    with open(filepath) as fi:
        claims = {
            int(k): Claim.from_dict(v)
            for k, v in tqdm(
                json.load(fi).items(), desc="loading claims", disable=not verbose
            )
        }

    if as_list:
        return sorted(list(claims.values()), key=lambda x: x.id)
    return claims


def save_claims(claims, filepath, **kwargs):
    with open(filepath, "w") as fo:
        json.dump({k: v.to_dict() for k, v in claims.items()}, fo, **kwargs)


def load_articles(filepath, as_list=False, verbose=True):
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


def save_articles(articles, filepath, **kwargs):
    with open(filepath, "w") as fo:
        json.dump({k: v.to_dict() for k, v in articles.items()}, fo, **kwargs)


def claims_from_phase1(metadata_file):
    with open(metadata_file) as fi:
        claims = {
            d["id"]: Claim(**d)
            for d in tqdm(json.load(fi), desc="loading claims from phase1")
        }

    for claim in claims.values():
        if not claim.related_articles:
            raise ValueError("claim {} has no related articles".format(claim.id))

        related_articles_dict = {}
        for rel_art in claim.related_articles:
            rel_art = str(rel_art) + ".txt"  # use full filename as id
            related_articles_dict[rel_art] = None
        claim.related_articles = related_articles_dict

    return claims


def _articles_from_phase1_visit(fpath):
    with open(fpath) as fi:
        art_id = os.path.basename(fpath)
        article = Article.from_txt(art_id, fi.read())
    return art_id, article


def articles_from_phase1(articles_dir, nproc=1):
    # note 1994 articles exists in the phase1 dataset that aren't a related
    # article for any of the claims, but we keep them as it's extra data we
    # may use in a corpus or something
    fpaths = glob.glob(os.path.join(articles_dir, "*.txt"))

    pool = multiprocessing.Pool(nproc)
    articles = {}
    for art_id, article in tqdm(
        pool.imap_unordered(_articles_from_phase1_visit, fpaths),
        total=len(fpaths),
        desc="loading articles from phase1",
    ):
        articles[art_id] = article

    return articles


def claims_from_phase2(metadata_file):
    with open(metadata_file) as fi:
        claims = {
            d["id"]: Claim(**d)
            for d in tqdm(json.load(fi), desc="loading claims from phase2")
        }

    # remove "train_articles" string from related_articles keys
    for claim in claims.values():
        if not claim.related_articles:
            continue
        keys = list(claim.related_articles.keys())
        for old_name in keys:
            new_name = os.path.basename(old_name)
            claim.related_articles[new_name] = claim.related_articles.pop(old_name)

        for key in claim.related_articles:
            assert "train" not in key

    return claims


def _articles_from_phase2_visit(_input):
    art_id, fpath, url = _input
    with open(fpath) as fi:
        article = Article.from_html(art_id, fi.read(), url=url)
    return art_id, article


def articles_from_phase2(articles_dir, claims, nproc=1):
    art_id_to_url = {
        k: v for claim in claims.values() for k, v in claim.related_articles.items()
    }

    _inputs = []
    fpaths = glob.glob(os.path.join(articles_dir, "*.html"))
    for fpath in fpaths:
        # certain articles are not part of any of the related articles from the claims:
        # articles = load all articles from articles dir
        # claims_articles_set = set([art for claim in claims for art in list(claim.related_articles.keys())])#
        # articles_set = set(articles.keys())
        # articles_set - claims_articles_set
        # the above produces 38 entries
        #
        # if the article is not found in art_id_to_url, this means none of the claims
        # ever refer it, so we ignore it
        try:
            art_id = os.path.basename(fpath)
            url = art_id_to_url[art_id]
            _inputs.append((art_id, fpath, url))
        except KeyError:
            continue

    pool = multiprocessing.Pool(nproc)
    articles = {}
    for art_id, article in tqdm(
        pool.imap_unordered(_articles_from_phase2_visit, _inputs),
        total=len(fpaths),
        desc="loading article from phase2",
    ):
        if art_id is not None:
            articles[art_id] = article

    return articles


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


def trim_metadata_phase1(claims_file, articles_dir, output_dir, n_examples):
    os.makedirs(os.path.join(output_dir, "articles"))

    # load data
    with open(claims_file, "r") as fi:
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


def trim_metadata_phase2(claims_file, articles_dir, output_dir, n_examples=None):
    os.makedirs(os.path.join(output_dir, "articles"))

    # load data
    with open(claims_file, "r") as fi:
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
    claims_file, articles_dir, train_dir, test_dir, train_size, random_state
):
    from sklearn.model_selection import train_test_split

    if os.path.exists(train_dir):
        raise ValueError("train_dir ({}) already exists".format(train_dir))
    if os.path.exists(test_dir):
        raise ValueError("test_dir ({}) already exists".format(test_dir))
    os.makedirs(os.path.join(train_dir, "articles"))
    os.makedirs(os.path.join(test_dir, "articles"))

    with open(claims_file, "r") as fi:
        metadata = json.load(fi)

    # log args
    _logger.info("claims_file:  %s", claims_file)
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
