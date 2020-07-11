import os
import json
import heapq
import random
import argparse
import multiprocessing

import spacy
import numpy as np
from tqdm.auto import tqdm

from valerie import search
from valerie.data import Article
from valerie.datasets import Phase2Dataset
from valerie.utils import get_logger
from valerie.modeling import (
    ClaimantModel,
    SequenceClassificationExample,
    SequenceClassificationModel,
)
from valerie.preprocessing import clean_text

_logger = get_logger()

# global resources
label_set = {0, 1, 2}
id2label = {0: "false", 1: "partly true", 2: "true"}
label2id = {"false": 0, "partly true": 1, "true": 2}
number2place = {1: "first", 2: "second"}
furthermore_syns = [
    "In addition",
    "Furthermore",
    "Consequently",
    "Additionaly",
    "Moreover",
]

_logger.info("... loading spacy ...")
nlp = spacy.load("en_core_web_lg")


########## Compile Final Output ##########


def compile_final_output(claims, seq_clf_predictions, claimant_predictions, support):
    output = {}
    for claim in claims:
        # label (predicted)
        pred = seq_clf_predictions[claim.id]
        assert isinstance(pred, int)
        assert pred in label_set

        # related articles
        related_articles = list(claim.related_articles.keys())
        related_articles = related_articles[:2]
        related_articles = {i + 1: x for i, x in enumerate(related_articles)}
        assert isinstance(related_articles, dict)
        assert len(related_articles) >= 0 and len(related_articles) <= 2
        for k, v in related_articles.items():
            assert k in {1, 2}
            assert isinstance(v, str)

        # explanation
        explanation = []

        first_source = None
        try:
            for sup in support[claim.id]:
                for i, rel_art in related_articles.items():
                    if (
                        len(explanation) >= 2
                        or sup["art_id"] != rel_art
                        or sup["similarity"] < 0.80
                    ):
                        continue

                    sup_text = sup["text"]
                    if len(sup_text) > 340:
                        sup_text = sup_text[:335] + " ..."

                    art = articles[rel_art]
                    if len(explanation) == 0:
                        explanation.append(
                            "The claim is {}, as explained in the {} "
                            'article, which states "{}".'.format(
                                id2label[pred], number2place[i], sup_text,
                            )
                        )
                        first_source = art.source
                    elif art.source == first_source and first_source is not None:
                        explanation.append(
                            '{}, the article goes on to say "{}".'.format(
                                random.choice(furthermore_syns), sup_text,
                            )
                        )
                    else:
                        explanation.append(
                            "This conclusion is also confirmed by the {} article{}, "
                            '"{}".'.format(
                                number2place[i],
                                " from " + art.source if art.source else "",
                                sup_text[:400],
                            )
                        )
        except:
            explanation = []

        # backup default explanation
        if not explanation:
            explanation.append(seq_clf_explanations[claim.id])

        if claimant_predictions[claim.id] == pred:
            explanation.append(claimant_explanations[claim.id])

        explanation = [e for e in explanation if e]
        explanation = " ".join(explanation)
        if len(explanation) > 995:
            explanation = explanation[:995] + " ..."
        explanation = explanation[:999]
        assert isinstance(explanation, str)
        assert len(explanation) < 1000

        # final predictions
        output[claim.id] = {
            "label": pred,
            "related_articles": related_articles,
            "explanation": explanation,
        }
    return output


########## Claimant Classification ##########


def claimant_classification(claims, claimant_model_file):
    claimant_model = ClaimantModel.from_pretrained(claimant_model_file)

    predictions = {}
    explanations = {}
    for claim in tqdm(claims, desc="ClaimantModel predictions"):
        pred = claimant_model.predict(claim)
        if pred is None:
            predictions[claim.id] = int(random.randint(0, 2))
            explanations[claim.id] = ""
            continue

        pred = int(np.argmax(pred))
        assert pred in label_set, pred
        predictions[claim.id] = pred

        if pred == 0:
            explanations[claim.id] = (
                "This is also supported by the fact that the claimant has a history "
                "of misinformation, having previously made {} (out of {}) "
                "false statements.".format(
                    claimant_model.model[claim.claimant]["false"],
                    claimant_model.model[claim.claimant]["total"],
                )
            )
        elif pred == 1:
            explanations[claim.id] = (
                "This is also supported by the fact that the claimant has a mixed "
                "record, having previously made {} (out of {}) partially "
                "correct/incorrect statements.".format(
                    claimant_model.model[claim.claimant]["partly"],
                    claimant_model.model[claim.claimant]["total"],
                )
            )
        else:
            explanations[claim.id] = (
                "This is also supported by the fact that the claimant has a good "
                "track record, having previously made "
                "{} (out of {}) factual statements.".format(
                    claimant_model.model[claim.claimant]["true"],
                    claimant_model.model[claim.claimant]["total"],
                )
            )

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


def generate_sequence_classification_examples(claims):
    examples = []
    for claim in tqdm(claims, desc="generating examples"):
        examples.append(
            SequenceClassificationExample(
                guid=claim.id,
                text_a=claim.claim,
                text_b=(claim.claimant if claim.claimant else "no claimant")
                + " "
                + (claim.date.split()[0] if claim.date else "no date"),
                label=claim.label,
            )
        )
    return examples


def sequence_classification(
    claims, pretrained_model_name_or_path, predict_batch_size, nproc
):
    examples = generate_sequence_classification_examples(claims)

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
        assert pred in label_set, pred
        predictions[example.guid] = pred

        if pred == 0:
            explanations[example.guid] = (
                "The AI model detected patterns in the claim consitent with "
                "misinformation."
            )
        elif pred == 1:
            explanations[example.guid] = (
                "The AI model detected patterns in the claim consitent with "
                "clickbait and/or partial truth."
            )
        else:
            explanations[example.guid] = (
                "The AI model couldn't detect any patterns in the claim "
                "consitent with misinformation, clickbait, disinformation, or "
                "fake news, suggesting that the claim is likely true."
            )

    return predictions, explanations


########## Generate Examples ##########


def _generate_support(claim, n_examples=4):
    claim_text = claim.claim
    if claim.claimant is not None:
        claim_text += claim.claimant
    claim_doc = nlp(claim_text, disable=["textcat", "tagger", "parser", "ner"],)

    support = []
    for k in claim.related_articles:
        article = articles[k]
        if not article.content:
            continue
        _title = article.title if article.title else ""
        article_doc = nlp(
            _title + article.content, disable=["textcat", "tagger", "ner"],
        )

        for sentence in article_doc.sents:
            if not np.count_nonzero(sentence.vector):
                continue
            support.append(
                {
                    "similarity": claim_doc.similarity(sentence),
                    "art_id": article.id,
                    "text": sentence.text,
                }
            )
    support = heapq.nlargest(n_examples, support, key=lambda x: x["similarity"])
    return claim, support


def generate_support(claims, nproc):
    pool = multiprocessing.Pool(nproc)

    all_support = {}
    for claim, support in tqdm(
        pool.imap_unordered(_generate_support, claims),
        total=len(claims),
        desc="generating support",
    ):
        all_support[claim.id] = support
    return all_support


########## Related Articles ##########


def select_related_articles(responses, article_limit=2):
    # atm, the most succesful strategy for picking related articles
    # scored by the automated algorithm is using only the api score
    # (we select top article_limit articles since the response hits are sorted
    # by the api score)
    all_articles = {}

    for res in tqdm(responses, desc="selecting articles"):
        claim = res["claim"]
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
            all_articles[hit["article"].id] = hit["article"]
            claim.related_articles[hit["article"].id] = hit["article"]

    return all_articles


########## Search ##########


# from query.py
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
        query += " " + claim.date.split(" ")[0].split("T")[0]

    if claim.claimant:
        query += " " + claim.claimant

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

        assert article.url == article.id
        output.append({"score": hit["score"], "article": article, "url": hit["url"]})
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
    responses = []
    for claim, query, res in tqdm(
        pool.imap_unordered(search_pipeline, claims),
        total=len(claims),
        desc="fetching query responses",
    ):
        responses.append({"claim": claim, "res": res, "query": query})
    return responses


########## Main ##########


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Main pipeline for the phase2 submission.")
    parser.add_argument("--metadata_file", type=str)
    parser.add_argument("--predictions_file", type=str)
    parser.add_argument("--claimant_model_file", type=str)
    parser.add_argument("--pretrained_model_name_or_path", type=str)
    parser.add_argument("--predict_batch_size", type=int, default=1)
    parser.add_argument("--nproc", type=int, default=1)
    args = parser.parse_args()

    # read metadata and convert to claims objects
    _logger.info("... reading claims from {} ...".format(args.metadata_file))
    claims = Phase2Dataset.from_raw(args.metadata_file).claims

    # fetch search api responses
    _logger.info("... fetching query responses ...")
    responses = get_responses(claims, args.nproc)

    # select related articles
    _logger.info("... selecting related articles ...")
    articles = select_related_articles(responses)

    # generate supporting article evidence for each claim
    _logger.info("... generating support ...")
    try:
        support = generate_support(claims, nproc=args.nproc)
    except:
        support = {}

    # get sequence classification predictions and explanations
    _logger.info("... generating sequence classification predictions ...")
    # print(f"\n\n\n\n{len(claims)}\nHEEYYYY IM HERE\n\n\n\n\n\n\n")
    seq_clf_predictions, seq_clf_explanations = sequence_classification(
        claims, args.pretrained_model_name_or_path, args.predict_batch_size, args.nproc,
    )

    # get claimant classification predictions and explanations
    _logger.info("... generating claimant predictions ...")
    claimant_predictions, claimant_explanations = claimant_classification(
        claims, args.claimant_model_file
    )

    # compile final output
    _logger.info("... compiling output ...")
    output = compile_final_output(
        claims, seq_clf_predictions, claimant_predictions, support
    )

    # write output to json
    _logger.info("... writing predictions to {} ...".format(args.predictions_file))
    with open(args.predictions_file, "w") as fo:
        json.dump(output, fo, indent=2)
    if not os.path.exists(args.predictions_file):
        raise ValueError("predictions file was not created")

    # done :)
    _logger.info("... done ...")
