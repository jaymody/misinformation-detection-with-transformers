"""Data classes."""
import os
import glob
import json
import logging
import multiprocessing

import bs4
import tldextract
from tqdm import tqdm

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
        self.claimant = claimant
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
        text = clean_text(text)
        return cls(id, content=text, **kwargs)

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


def save_claims(filepath, claims, **kwargs):
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


def save_articles(filepath, articles, **kwargs):
    with open(filepath, "w") as fo:
        json.dump({k: v.to_dict() for k, v in articles.items()}, fo, **kwargs)


def claims_from_phase2(claims_file):
    with open(claims_file) as fi:
        claims = {
            d["id"]: Claim(**d) for d in tqdm(json.load(fi), desc="loading claims")
        }

    # remove "train_articles" string from related_articles keys
    for claim in claims.values():
        keys = list(claim.related_articles.keys())
        for old_name in keys:
            new_name = os.path.basename(old_name)
            claim.related_articles[new_name] = claim.related_articles.pop(old_name)

    for claim in claims.values():
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
        desc="loading articles",
    ):
        if art_id is not None:
            articles[art_id] = article

    return articles
