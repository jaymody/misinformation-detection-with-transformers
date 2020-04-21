"""Defines structs for high-level data types."""
import os
import logging

import bs4

from . import utils

_logger = logging.getLogger(__name__)


class Claim:
    """A claim."""

    def __init__(self, id, text, claimant=None, label=None, date=None, related_articles=None, support=None):
        """Constructor for Claim."""
        self.id = id
        self.text = utils.clean_text(text)
        self.claimant = claimant
        self.label = label
        self.data = date
        self.related_articles = related_articles
        self.support = support


class Article:
    """An article."""

    def __init__(self, id, body=None, title=None, source=None, author=None, url=None, date=None):
        """Constructor for Article."""
        self.id = id
        self.title = title
        self.body = body
        self.source = source
        self.date = date

    @classmethod
    def from_txt(cls, filepath, **kwargs):
        """Construct an Article given a text file."""
        if "id" not in kwargs:
            id = os.path.splitext[os.path.basename(filepath)][0]

        with open(filepath, 'r') as fi:
            text = utils.clean_text(fi.read())
            return cls(id, body=text, **kwargs)

    @classmethod
    def from_html(cls, filepath, **kwargs):
        """Constructs an Article given an html file."""
        if "id" not in kwargs:
            id = os.path.splitext[os.path.basename(filepath)][0]

        with open(filepath, 'r') as fi:
            html = fi.read()

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
            t = utils.clean_text(t)
            if t and len(t) > 32: # dissallow empty/short text sequences
                text += t + " "

        title = utils.clean_text(soup.title.string) if soup.title else None
        return cls(id, body=text, title=title , **kwargs)
