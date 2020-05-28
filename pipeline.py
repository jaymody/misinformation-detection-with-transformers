import os
import json
import random
import argparse
import multiprocessing

import spacy
import numpy as np
from tqdm import tqdm

from valerie import search
from valerie.data import claims_from_phase2, Article
from valerie.utils import get_logger
from valerie.modeling import (
    ClaimantModel,
    SequenceClassificationExample,
    SequenceClassificationModel,
)
from valerie.preprocessing import clean_text

_logger = get_logger()

_logger.info("... loading spacy ...")
nlp = spacy.load("en_core_web_lg")


########## Claimant Classification ##########


def claimant_classification(claims, claimant_model_file):
    claimant_model = ClaimantModel.from_pretrained(claimant_model_file)

    predictions = {}
    explanations = {}
    for k, claim in tqdm(claims.items(), desc="ClaimantModel predictions"):
        pred = claimant_model.predict(claim)
        if pred is None:
            predictions[k] = int(random.randint(0, 2))
            explanations[k] = "Claimant not found in ClaimantModel."
        else:
            pred = int(np.argmax(pred))
            predictions[k] = pred
            if pred == 0:
                explanations[k] = "The claimant has a record of lying."
            elif pred == 1:
                explanations[k] = "The claimant has mixed record."
            else:
                explanations[k] = "The claimant has a good record."
            explanations[k] += json.dumps(claimant_model.model[claim.claimant])

    return predictions, explanations


########## Sequence Classification ##########


def _sequence_classification(
    examples, pretrained_model_name_or_path, predict_batch_size, nproc,
):
    model = SequenceClassificationModel.from_pretrained(pretrained_model_name_or_path)

    predict_dataset = model.create_dataset(examples=examples, nproc=nproc)
    prediction_output = model.predict(predict_dataset, predict_batch_size)

    # returns an array of probs
    probabilities = prediction_output.predictions
    return probabilities


def sequence_classification(
    examples, pretrained_model_name_or_path, predict_batch_size, nproc
):
    probabilities = _sequence_classification(
        examples,
        pretrained_model_name_or_path,
        predict_batch_size=predict_batch_size,
        nproc=nproc,
    )

    if len(probabilities) != len(examples):
        raise ValueError(
            "len predictions ({}) != len examples ({})".format(
                len(probabilities), len(examples)
            )
        )

    predictions = {}
    explanations = {}
    for example, prob in zip(examples, probabilities):
        pred = int(np.argmax(prob))
        predictions[example.guid] = pred
        explanations[example.guid] = "Transformer model output = {}.".format(pred)

    return predictions, explanations


########## Generate Examples ##########


def generate_examples(claims):
    examples = []

    for k, claim in tqdm(claims.items(), desc="generating examples"):
        examples.append(
            SequenceClassificationExample(
                guid=k, text_a=claim.claim, label=claim.label,
            )
        )

    return examples


########## Related Articles ##########


def select_related_articles(claims, responses, article_limit=2):
    # currently, articles are sorted by api score
    # atm, the most succesful strategy for picking related articles
    # scored by the automated algorithm is using only the api score
    # (we select top article_limit articles since the response hits are sorted
    # by the api score)
    all_articles = {}

    for k, res in tqdm(responses.items(), desc="selecting articles"):
        claim = claims[k]
        claim.related_articles = {}
        if res["res"] is None:
            continue

        cur_art = 0
        for hit in res["res"]["hits"]["hits"]:
            if cur_art >= article_limit:
                break
            if not hit["article"]:
                continue
            cur_art += 1
            all_articles[hit["article"].url] = hit["article"]
            claim.related_articles[hit["article"].url] = hit["article"]

    return all_articles


########## Search ##########


# from query.py
def query_expansion(claim):
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


# from query.py
def convert_html_hits_to_article(res):
    visited = set()
    output = []

    for hit in res["hits"]["hits"]:
        if hit["url"] in visited:
            continue

        article = Article.from_html(hit["url"], hit["content"], url=hit["url"])
        if not article.content or len(article.content) < 32:
            continue

        # note we don't convert to dict here
        output.append({"score": hit["score"], "article": article})

        # not sure why id is not always being copied to dict, so here's a fix
        output[-1]["article"].id = hit["url"]

        visited.add(hit["url"])

    return output


# from query.py
def search_pipeline(claim):
    query = query_expansion(claim)
    res = search.query(query)
    if res is not None:
        res["hits"]["hits"] = convert_html_hits_to_article(res)
    return claim, query, res


def get_responses(claims, nproc):
    pool = multiprocessing.Pool(nproc)
    responses = {}
    for claim, query, res in tqdm(
        pool.imap_unordered(search_pipeline, claims.values()),
        total=len(claims),
        desc="fetching query responses",
    ):
        responses[claim.id] = {"id": claim.id, "res": res, "query": query}
    return responses


if __name__ == "__main__":
    # load claims
    # run queries
    # choose related articles
    # make predictions
    # generate sequence classification examples
    #   generate sequence classification predictions
    #   generate claimant predictions
    #   combine predictions for final predictions
    #   form explanation (the claimant has a history of lying... this sentence
    #   from the following article states ... which means ... see x article)
    # write predictions to file

    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata_file", type=str)
    parser.add_argument("--predictions_file", type=str)
    parser.add_argument("--claimant_model_file", type=str)
    parser.add_argument("--pretrained_model_name_or_path", type=str)
    parser.add_argument("--predict_batch_size", type=int, default=1)
    parser.add_argument("--nproc", type=int, default=1)
    args = parser.parse_args()

    _logger.info("... reading claims from {} ...".format(args.metadata_file))
    claims = claims_from_phase2(args.metadata_file)

    _logger.info("... fetching query responses ...")
    responses = get_responses(claims, args.nproc)

    _logger.info("... selecting related articles ...")
    articles = select_related_articles(claims, responses)

    examples = generate_examples(claims)
    seq_clf_predictions, seq_clf_explanations = sequence_classification(
        examples,
        args.pretrained_model_name_or_path,
        args.predict_batch_size,
        args.nproc,
    )

    _logger.info("... generating claimant predictions ...")
    claimant_predictions, claimant_explanations = claimant_classification(
        claims, args.claimant_model_file
    )

    _logger.info("... compiling output ...")
    output = {}
    for k, claim in claims.items():
        pred = seq_clf_predictions[k]
        assert isinstance(pred, int)
        assert pred in [0, 1, 2]

        explanation = seq_clf_explanations[k]
        if claimant_predictions[k] == pred:
            explanation += claimant_explanations[k]
        explanation = explanation[:999]
        assert isinstance(explanation, str)
        assert len(explanation) < 1000

        related_articles = list(claim.related_articles.keys())
        related_articles = related_articles[:2]
        assert isinstance(related_articles, list)
        assert len(related_articles) >= 0 and len(related_articles) <= 2
        for rel_art in related_articles:
            assert isinstance(rel_art, str)

        output[k] = {
            "label": pred,
            "related_articles": related_articles,
            "explanation": explanation,
        }

    _logger.info("... writing predictions to {} ...".format(args.predictions_file))
    with open(args.predictions_file, "w") as fo:
        json.dump(output, fo, indent=2)

    if not os.path.exists(args.predictions_file):
        raise ValueError("predictions file was not created")
    _logger.info("... done ...")
