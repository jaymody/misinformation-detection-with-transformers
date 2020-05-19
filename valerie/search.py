import os
import logging

import nltk
import requests

_logger = logging.getLogger(__name__)


if "LEADERS_PRIZE_API_KEY" not in os.environ:
    raise ValueError("LEADERS_PRIZE_API_KEY not found in environment variables.")

API_KEY = os.environ["LEADERS_PRIZE_API_KEY"]
SEARCH_API_URL = "http://lpsa.wrw.org/claimserver/api/v1.0/evidence"


def query(query_string, from_idx=0):
    resp = requests.get(
        SEARCH_API_URL,
        headers={"X-Api-Key": API_KEY},
        params={"query": query_string, "from": from_idx},
    )
    if resp.status_code == requests.codes.ok:  # pylint: disable=no-member
        return resp.json()
    else:
        _logger.warning("query failed, response status is not ok")


def enhanced_query(query_string, stopwords, from_idx=0):
    words = nltk.tokenize.word_tokenize(query_string)
    words = [word for word in words if word not in stopwords]
    return query(" ".join(words), from_idx=from_idx)
