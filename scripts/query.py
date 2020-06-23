import json
import argparse
import multiprocessing

import spacy
from tqdm import tqdm

from valerie import search
from valerie.data import Article, load_claims
from valerie.utils import get_logger
from valerie.scoring import validate_predictions_phase2, compute_score_phase2
from valerie.preprocessing import clean_text

_logger = get_logger()

nlp = spacy.load("en_core_web_lg")


def compute_query_score(responses, claims):
    predictions = {}
    perfect_predictions = {}
    labels = {}

    for k, v in responses.items():
        claim = claims[k]
        labels[claim.id] = claim.to_dict()
        predictions[claim.id] = {
            "label": claim.label,
            "related_articles": [
                hit["article"]["url"] for hit in v["res"]["hits"]["hits"][:2]
            ]
            if v["res"]
            else [],
            "explanation": "",
        }
        perfect_predictions[claim.id] = {
            "label": claim.label,
            "related_articles": [
                hit["article"]["url"]
                for hit in v["res"]["hits"]["hits"]
                if hit["article"]["url"] in claim.related_articles.values()
            ][:2]
            if v["res"]
            else [],
            "explanation": "",
        }

    validate_predictions_phase2(predictions)
    score = compute_score_phase2(labels, predictions)
    validate_predictions_phase2(perfect_predictions)
    perfect_score = compute_score_phase2(labels, perfect_predictions)
    return {
        "perfect_score": perfect_score["score"],
        "perfect_error": perfect_score["error"],
        "api_score": score["score"],
        "api_error": score["error"],
    }


def generate_query(claim):
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

        output.append({"score": hit["score"], "article": article.to_dict()})

        # not sure why id is not always being copied to dict, so here's a fix
        output[-1]["article"]["id"] = hit["url"]

        visited.add(hit["url"])

    return output


def pipeline(claim):
    query = generate_query(claim)
    res = search.query(query)
    res["hits"]["hits"] = convert_html_hits_to_article(res)
    return claim, query, res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file", type=str)
    parser.add_argument(
        "--claims_file", type=str, default="data/phase2/all-data/claims.json"
    )
    parser.add_argument("--nproc", type=int, default=4)
    args = parser.parse_args()

    claims = load_claims(args.claims_file)
    run_claims = list(claims.values())

    pool = multiprocessing.Pool(args.nproc)
    responses = {}
    for claim, query, res in tqdm(
        pool.imap_unordered(pipeline, run_claims), total=len(run_claims)
    ):
        responses[claim.id] = {"id": claim.id, "res": res, "query": query}

    with open(args.output_file, "w") as fo:
        json.dump(responses, fo, indent=2)

    _logger.info("Missed Queries: %d", sum(1 for v in responses.values() if v is None))
    _logger.info(
        "Scores: %s", json.dumps(compute_query_score(responses, claims), indent=2)
    )
