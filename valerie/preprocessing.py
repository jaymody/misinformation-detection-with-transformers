"""Utility functions."""
import re
import nltk
import logging
import unicodedata

import wordninja

_logger = logging.getLogger(__name__)


def extract_words_from_url(url):
    """Extracts words from a url.

    Example
    -------
    input:  https://www.berkeleyschools.net/departments/public-information-office/
    output: berkeley schools departments public information office
    """
    remove = {
        "www",
        "html",
        "index",
        "htm",
        "http:",
        "https:",
        "http",
        "https",
        "com",
        "ca",
        "gov",
        "org",
        "net",
        "co",
    }
    words = [
        clean_text(w, remove_punctuation=True)
        for w in split(url, [".", "-", "/", "?", "=", "&"])
    ]
    words = [w for word in words for w in wordninja.split(word)]
    words = [
        word
        for word in words
        if word and word not in remove and not word.isnumeric() and len(word) > 2
    ]
    return words


def split(string, delimiters):
    """Split a string using multiple delimiters."""
    regexPattern = "|".join(map(re.escape, delimiters))
    return re.split(regexPattern, string)


def split_sentences(text):
    """Returns the input text split into a list of sentences."""
    return [sentence.strip() for sentence in nltk.tokenize.sent_tokenize(text)]


def clean_text(text, remove_punctuation=False):
    """Cleans the text of whitespace and control chars."""
    output = []
    for char in text:
        cp = ord(char)
        if cp == 0 or cp == 0xFFFD or _is_control(char):
            continue
        if remove_punctuation and _is_punctuation(char):
            continue
        if _is_whitespace(char):
            output.append(" ")
        else:
            output.append(char)
    return "".join(output).strip()


def _is_punctuation(char):
    cp = ord(char)
    if (
        (cp >= 33 and cp <= 47)
        or (cp >= 58 and cp <= 64)
        or (cp >= 91 and cp <= 96)
        or (cp >= 123 and cp <= 126)
    ):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


def _is_whitespace(char):
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char):
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat in ("Cc", "Cf"):
        return True
    return False
