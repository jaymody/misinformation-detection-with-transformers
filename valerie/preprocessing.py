"""Text preprocessing."""
import logging
import unicodedata

import nltk

_logger = logging.getLogger(__name__)


def split_sentences(text):
    """Returns the input text split into a list of sentences."""
    return [sentence.strip() for sentence in nltk.tokenize.sent_tokenize(text)]


def clean_text(text):
    """Cleans the text of whitespace and control chars."""
    output = []
    for char in text:
        cp = ord(char)
        if cp == 0 or cp == 0xfffd or _is_control(char):
            continue
        if _is_whitespace(char):
            output.append(" ")
        else:
            output.append(char)
    return "".join(output).strip()


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
