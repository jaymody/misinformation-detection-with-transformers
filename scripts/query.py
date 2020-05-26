import os
import json
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
    score = compute_score_phase2(predictions, labels)
    validate_predictions_phase2(perfect_predictions)
    perfect_score = compute_score_phase2(perfect_predictions, labels)
    return {
        "perfect_score": perfect_score["score"],
        "perfect_error": perfect_score["error"],
        "api_score": score["score"],
        "api_error": score["error"],
    }


def generate_query(claim):
    claim_doc = nlp(claim.claim, disable=["textcat", "tagger", "parser"])

    # stopword removal
    query_words = [token.text for token in claim_doc if not token.is_stop]

    # add claimant to query
    if claim.claimant:
        query_words.insert(0, claim.claimant)

    # pad query with named entities
    if len(query_words) < 8:
        query_words += [ent.text for ent in claim_doc.ents]

    query = clean_text(" ".join(query_words), remove_punctuation=True)
    return query


def convert_html_hits_to_article(res):
    visited = set()
    output = []

    for hit in res["hits"]["hits"]:
        if hit["url"] in visited:
            continue

        article = Article.from_html(hit["url"], hit["content"], url=hit["url"])
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
    claims = load_claims("data/phase2/all_data/claims.json")
    run_claims = list(claims.values())

    pool = multiprocessing.Pool(4)
    responses = {}
    for claim, query, res in tqdm(
        pool.imap_unordered(pipeline, run_claims), total=len(run_claims)
    ):
        responses[claim.id] = {"id": claim.id, "res": res, "query": query}

    with open("data/phase2/queries.json", "w") as fo:
        json.dump(responses, fo, indent=2)

    print("Missed Queries:", sum(1 for v in responses.values() if v is None))
    print("Scores:", json.dumps(compute_query_score(responses, claims), indent=2))
