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
from valerie.modeling import ClaimantModel
from valerie.preprocessing import clean_text

_logger = get_logger()

_logger.info("... loading spacy ...")
nlp = spacy.load("en_core_web_lg")


########## Claimant Classification ##########


def claimant_classification(claims, claimant_model_file):
    claimant_model = ClaimantModel.from_pretrained(claimant_model_file)

    predictions = {}
    explanations = {}
    for k, claim in tqdm(claims.items(), desc="ClaimantModel Predictions"):
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

# # from train.py
# def get_args_files(output_dir):
#     if not os.path.exists(output_dir):
#         raise ValueError(f"output dir does not exists: {output_dir}")

#     args_files = {
#         "config_args_file": os.path.join(output_dir, "config_args.json"),
#         "tokenizer_args_file": os.path.join(output_dir, "tokenizer_args.json"),
#         "model_args_file": os.path.join(output_dir, "model_args.json"),
#     }
#     for k, v in args_files.items():
#         if not os.path.exists(v):
#             raise ValueError(f"could not find {k}: {v}")

#     return args_files


# def _sequence_classification(
#     examples,
#     pretrained_model_name_or_path,
#     config_args_file,
#     tokenizer_args_file,
#     model_args_file,
#     predict_batch_size,
#     nproc,
# ):
#     config_args = {}
#     tokenizer_args = {}
#     model_args = {}
#     if config_args_file:
#         with open(config_args_file) as fi:
#             config_args = json.load(fi)
#     if tokenizer_args_file:
#         with open(tokenizer_args_file) as fi:
#             tokenizer_args = json.load(fi)
#     if model_args_file:
#         with open(model_args_file) as fi:
#             model_args = json.load(fi)

#     with open(data_args_file) as fi:
#         data_args = DataArguments(**json.load(fi))
#     with open(training_args_file) as fi:
#         training_args = SequenceClassificationTrainingArgs(
#             output_dir=output_dir, logging_dir=output_dir, **json.load(fi)
#         )

#     model = SequenceClassificationModel.from_pretrained(
#         pretrained_model_name_or_path, config_args, tokenizer_args, model_args
#     )

#     predict_dataset = model.create_dataset(examples=examples, nproc=nproc)
#     predictions = model.predict(predict_dataset, predict_batch_size)


# def sequence_classification(
#     examples, pretrained_model_name_or_path, predict_batch_size, nproc
# ):
#     args_files = get_args_files(pretrained_model_name_or_path)
#     _predict(
#         examples,
#         pretrained_model_name_or_path,
#         **args_files,
#         predict_batch_size=predict_batch_size,
#         nproc=nproc,
#     )

#     # returns a dict of claim ids with their associated classification output probs/features


########## Create Examples ##########


# def create_examples(claims, articles):
#     pass


########## Related Articles ##########


def select_related_articles(claims, responses, article_limit=2):
    # currently, articles are sorted by api score
    # atm, the most succesful strategy for picking related articles
    # scored by the automated algorithm is using only the api score
    # (we select top article_limit articles since the response hits are sorted
    # by the api score)
    all_articles = {}

    for k, res in tqdm(responses.items(), desc="Selecting Articles"):
        claim = claims[k]
        claim.related_articles = {}

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
    res["hits"]["hits"] = convert_html_hits_to_article(res)
    return claim, query, res


def get_responses(claims, nproc):
    pool = multiprocessing.Pool(nproc)
    responses = {}
    for claim, query, res in tqdm(
        pool.imap_unordered(search_pipeline, claims.values()),
        total=len(claims),
        desc="Getting Query Responses",
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
    parser.add_argument("--nproc", type=int)
    args = parser.parse_args()

    _logger.info("... reading claims from {} ...".format(args.metadata_file))
    claims = claims_from_phase2(args.metadata_file)

    _logger.info("... getting query responses ...")
    responses = get_responses(claims, args.nproc)

    _logger.info("... selecting related articles ...")
    articles = select_related_articles(claims, responses)

    # examples = create_examples(claims, articles)
    # seq_clf_features = sequence_classification()

    _logger.info("... generating claimant predictions ...")
    predictions, explanations = claimant_classification(
        claims, args.claimant_model_file
    )

    _logger.info("... compiling output ...")
    output = {}
    for k, claim in claims.items():
        pred = predictions[k]
        assert isinstance(pred, int)
        assert pred in [0, 1, 2]

        explanation = explanations[k]
        explanation = explanation[:999]
        assert isinstance(explanation, str)

        related_articles = list(claim.related_articles.keys())
        related_articles = related_articles[:2]
        assert isinstance(related_articles, list)
        assert len(related_articles) >= 0 and len(related_articles) <= 2
        assert isinstance(related_articles[0], str)
        assert isinstance(related_articles[1], str)

        output[k] = {
            "label": predictions[k],
            "related_articles": related_articles,
            "explanation": explanation,
        }

    _logger.info("... writing predictions to {} ...".format(args.predictions_file))
    with open(args.predictions_file, "w") as fo:
        json.dump(output, fo, indent=2)

    if not os.path.exists(args.predictions_file):
        raise ValueError("predictions file was not created")
    _logger.info("... done ...")
