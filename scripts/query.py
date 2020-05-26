import os
import json
import multiprocessing

import nltk
import spacy
from tqdm import tqdm

from valerie import search
from valerie.data import Article
from valerie.utils import load_claims, get_logger
from valerie.scoring import validate_predictions_phase2, compute_score_phase2
from valerie.modeling import SourceModel
from valerie.preprocessing import extract_words_from_url, clean_text

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
            "related_articles": [hit["url"] for hit in v["res"]["hits"]["hits"][:2]]
            if v["res"]
            else [],
            "explanation": "",
        }
        perfect_predictions[claim.id] = {
            "label": claim.label,
            "related_articles": [
                hit["url"]
                for hit in v["res"]["hits"]["hits"]
                if hit["url"] in claim.related_articles.values()
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


def convert_html_to_text_content(res):
    visited = set()
    for i, hit in enumerate(res["hits"]["hits"]):
        if hit["url"] in visited:
            continue

        article = Article.from_html(hit["url"], hit["content"], url=hit["url"])
        res["hits"]["hits"][i]["content"] = article.content
        visited.add(hit["url"])


def pipeline(claim):
    query = generate_query(claim)
    res = search.query(query)
    convert_html_to_text_content(res)
    return claim, query, res


if __name__ == "__main__":
    claims = load_claims("data/phase2/all_data/claims.json", as_list=True)[:None]

    pool = multiprocessing.Pool(4)
    responses = {}
    for claim, query, res in tqdm(
        pool.imap_unordered(pipeline, claims), total=len(claims)
    ):
        responses[claim.id] = {"id": claim.id, "res": res, "query": query}

    with open("data/phase2/queries.json", "w") as fo:
        json.dump(responses, fo, indent=2)

    print("Missed Queries:", sum(1 for v in responses.values() if v is None))
    print("Scores:", json.dumps(compute_query_score(responses, claims), indent=2))
