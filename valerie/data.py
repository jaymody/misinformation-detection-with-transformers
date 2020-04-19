"""Defines structs for high-level data types."""
import logging

import bs4

from . import preprocessing

_logger = logging.getLogger(__name__)


class Article:
    def __init__(self, id, body=None, title=None, source=None, date=None):
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
    def __init__(self, id, url=None, author=None, **kwargs):
        self.url = url
        self.author = author
        super().__init__(id, **kwargs)

    @classmethod
    def from_html(cls, id, html, **kwargs):
        def tag_visible(element):
            whitelist = [
                "h1",
                "h2",
                "h3",
                "h4",
                "h5",
                "body",
                "p",
                "font"
            ]
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


    @classmethod
    def from_html_alternative(cls, id, html, **kwargs):
        soup = bs4.BeautifulSoup(html, "html.parser")

        blacklist = [
            "noscript",
            "input",
            "header",
            "style",
            "script",
            "head",
            "title",
            "meta",
            "[document]"
        ]
        for script in soup(blacklist):
            script.extract()

        text = soup.get_text()
        text = preprocessing.clean_text(text)

        title = preprocessing.clean_text(soup.title.string) if soup.title else None
        return cls(id, body=text, title=title , **kwargs)
