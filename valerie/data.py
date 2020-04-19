"""Defines structs for high-level data types."""
import logging

import bs4

from . import preprocessing

_logger = logging.getLogger(__name__)


class InputExample:
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructor for `InputExample`.

        Parameters
        ----------
        guid : int
            Unique id for the example.
        text_a : str
            The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
        text_b : str, optional
            The untokenized text of the second sequence. Only must be specified
            for sequence pair tasks.
        label : str, optional
            The label of the example. This should be specified for train and
            dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures:
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        """Constructor for `InputFeatures`."""
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class Article:
    """An article."""

    def __init__(self, id, body=None, title=None, source=None, date=None):
        """Constructor for Article."""
        self.id = id
        self.title = title
        self.body = body
        self.source = source
        self.date = date

    @classmethod
    def from_txt(cls, filepath, id, **kwargs):
        with open(filepath, 'r') as fi:
            text = preprocessing.clean_text(fi.read())
            return cls(id, body=text, **kwargs)


class WebArticle(Article):
    """An online article."""

    def __init__(self, id, url=None, author=None, **kwargs):
        """Constructor for WebArticle."""
        self.url = url
        self.author = author
        super().__init__(id, **kwargs)

    @classmethod
    def from_html(cls, id, html, **kwargs):
        """Create a WebArticle object from an html text file."""
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
            t = preprocessing.clean_text(t)
            if t and len(t) > 32: # dissallow empty/short text sequences
                text += t + " "

        title = preprocessing.clean_text(soup.title.string) if soup.title else None
        return cls(id, body=text, title=title , **kwargs)
