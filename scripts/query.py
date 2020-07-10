import json
import pickle
import argparse
import multiprocessing

import spacy
from tqdm.auto import tqdm

from valerie import search
from valerie.data import Article
from valerie.datasets import Phase2Dataset
from valerie.utils import get_logger
from valerie.scoring import validate_predictions_phase2, compute_score_phase2
from valerie.preprocessing import clean_text

_logger = get_logger()

nlp = spacy.load("en_core_web_lg")


def compute_responses_score(responses):
    predictions = {}
    perfect_predictions = {}
    labels = {}

    for v in responses:
        claim = v["claim"]
        labels[claim.id] = claim.to_dict()
        predictions[claim.id] = {
            "label": claim.label,
            "related_articles": {
                i + 1: x
                for i, x in enumerate(
                    [hit["url"] for hit in v["res"]["hits"]["hits"][:2]]
                )
            }
            if v["res"]
            else {},
            "explanation": "",
        }
        perfect_predictions[claim.id] = {
            "label": claim.label,
            "related_articles": {
                i + 1: x
                for i, x in enumerate(
                    [
                        hit["url"]
                        for hit in v["res"]["hits"]["hits"]
                        if hit["url"] in claim.related_articles.values()
                    ][:2]
                )
            }
            if v["res"]
            else {},
            "explanation": "",
        }

    validate_predictions_phase2(predictions)
    score = compute_score_phase2(labels, predictions)
    validate_predictions_phase2(perfect_predictions)
    perfect_score = compute_score_phase2(labels, perfect_predictions)
    return {
        "perfect_rerank_score": perfect_score["score"],
        "perfect_rerank_error": perfect_score["error"],
        "api_score": score["score"],
        "api_error": score["error"],
    }


def query_expansion(claim):
    claim_doc = nlp(claim.claim, disable=["textcat", "tagger", "parser", "ner"])

    # stopword removal
    query_words = [token.text for token in claim_doc if not token.is_stop]
    query = clean_text(
        " ".join(
            [
                t
                for t in query_words
                if t and not len(clean_text(t, remove_punctuation=True)) == 0
            ]
        )
    )

    if claim.date:
        query += " " + claim.date.split(" ")[0]

    if claim.claimant:
        query += " " + claim.claimant

    return query


def convert_html_hits_to_article(res):
    visited = set()
    output = []

    for hit in res["hits"]["hits"]:
        if hit["url"] in visited:
            continue

        article = Article.from_html(hit["url"], hit["content"], url=hit["url"])
        if not article.content or len(article.content) < 32:
            continue

        output.append({"score": hit["score"], "article": article, "url": hit["url"]})
        visited.add(hit["url"])

    return output


def pipeline(claim):
    query = query_expansion(claim)
    res = search.query(query)
    if res:
        res["hits"]["hits"] = convert_html_hits_to_article(res)
    return claim, query, res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--metadata_file", type=str)
    parser.add_argument("--nproc", type=int, default=4)
    args = parser.parse_args()

    claims = Phase2Dataset.from_raw(args.metadata_file).claims

    pool = multiprocessing.Pool(args.nproc)
    responses = []
    for claim, query, res in tqdm(
        pool.imap_unordered(pipeline, claims),
        total=len(claims),
        desc="fetching responses",
    ):
        responses.append({"claim": claim, "res": res, "query": query})

    with open(args.output_file, "wb") as fo:
        pickle.dump(responses, fo)

    _logger.info("Missed Queries: %d", sum(1 for v in responses if v["res"] is None))
    _logger.info("Scores: %s", json.dumps(compute_responses_score(responses), indent=2))
