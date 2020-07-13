import os
import json
import heapq
import random
import argparse
import collections
import multiprocessing

import spacy
import numpy as np
from tqdm.auto import tqdm

from valerie import search
from valerie.data import Article
from valerie.datasets import Phase2Dataset
from valerie.utils import get_logger, log_title
from valerie.modeling import (
    ClaimantModel,
    SequenceClassificationExample,
    SequenceClassificationModel,
)
from valerie.preprocessing import clean_text

_logger = get_logger(use_tqdm_handler=False)

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

log_title(_logger, "loading spacy")
nlp = spacy.load("en_core_web_lg")


########## Compile Final Output ##########


def compile_final_output(claims, seq_clf_predictions, claimant_predictions):
    output = {}
    for claim in claims:
        # label (predicted)
        pred = seq_clf_predictions[claim.id]
        assert isinstance(pred, int)
        assert pred in label_set

        # related articles
        related_articles = list(claim.related_articles.values())
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
            for sup in claim.support:
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

                    art = articles_dict[rel_art]
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
            predictions[claim.id] = -1
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
    examples, pretrained_model_name_or_path, predict_batch_size, nproc
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
                + (claim.date.split()[0].split("T")[0] if claim.date else "no date"),
                label=claim.label,
            )
        )
    return examples


def sequence_classification(
    claims, pretrained_model_name_or_path, predict_batch_size, nproc
):
    examples = generate_sequence_classification_examples(claims)
    _logger.info(
        "%s", json.dumps([example.__dict__ for example in examples[:5]], indent=2),
    )

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


########## Support ##########


def _generate_support(claim, n_examples=4):
    claim_text = claim.claim
    if claim.claimant is not None:
        claim_text += claim.claimant

    support = []
    for article_id in claim.related_articles.values():
        article = articles_dict[article_id]
        if not article.content:
            continue

        for sentence in article.doc.sents:
            if not np.count_nonzero(sentence.vector):
                continue
            support.append(
                {
                    "similarity": claim.doc.similarity(sentence),
                    "art_id": article.id,
                    "text": sentence.text,
                }
            )
    support = heapq.nlargest(n_examples, support, key=lambda x: x["similarity"])
    return claim, support


def generate_support(claims):
    support_dict = {}
    for claim in tqdm(claims, total=len(claims), desc="generating support",):
        claim, support = _generate_support(claim)
        support_dict[claim.id] = support

    return support_dict


########## Related Articles ##########


def get_related_articles(claims, keep_n_articles=2):
    article_dict = {}
    related_articles_dict = collections.defaultdict(dict)

    n_total_articles = 0
    n_empty_articles = 0
    n_chosen_articles = 0
    for claim in tqdm(claims, desc="selecting articles"):
        if claim.res is None:
            continue
        n_total_articles += len(claim.res["hits"]["hits"])

        cur_art = 0
        for hit in claim.res["hits"]["hits"]:  # hits are already sorted by api score
            article = hit["article"]

            # stop once we've reached the limit
            if cur_art >= keep_n_articles:
                break

            # this should never happen but just incase
            if not article or not article.content:
                n_empty_articles += 1
                continue
            article_dict[article.id] = article
            related_articles_dict[claim.id][cur_art + 1] = article.id

            # logging
            n_chosen_articles += 1
            cur_art += 1

    _logger.info("  max_total_hits:         %d", len(claims) * 30)
    _logger.info("  n_total_hits:           %d", n_total_articles)
    _logger.info("  max_articles_to_keep:   %d", len(claims) * keep_n_articles)
    _logger.info("  n_chosen_articles:      %d", n_chosen_articles)
    _logger.info("  n_empty_articles:       %d", n_empty_articles)

    return article_dict, related_articles_dict


def generate_article_docs(articles, nproc):
    def generate_doc_text(article):
        _title = article.title if article.title else ""
        return _title + article.content

    article_docs = {
        article.id: doc
        for article, doc in tqdm(
            zip(
                articles,
                nlp.pipe(
                    [generate_doc_text(article) for article in articles],
                    n_process=nproc,
                    disable=["textcat", "tagger", "ner"],
                ),
            ),
            total=len(articles),
            desc="running spacy on articles",
        )
    }
    return article_docs


########## Search ##########


def query_expansion(claim):
    # stopword removal
    query_words = [token.text for token in claim.doc if not token.is_stop]
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


def search_pipeline(_input):
    claim_id, query = _input
    res = search.query(query)
    if res is not None:
        res["hits"]["hits"] = convert_html_hits_to_article(res)
    return claim_id, query, res


def get_responses(claims, nproc):
    queries = {}
    for claim in tqdm(claims, desc="generating queries"):
        queries[claim.id] = query_expansion(claim)

    pool = multiprocessing.Pool(nproc)
    _inputs = [(k, query) for k, query in queries.items()]
    responses = {}
    for claim_id, query, res in tqdm(
        pool.imap_unordered(search_pipeline, _inputs),
        total=len(_inputs),
        desc="fetching query responses",
    ):
        queries[claim_id] = query
        responses[claim_id] = res

    return queries, responses


########## Claims ##########


def generate_claim_docs(claims, nproc):
    claim_docs = {
        claim.id: doc
        for claim, doc in tqdm(
            zip(
                claims,
                nlp.pipe(
                    [claim.claim for claim in claims],
                    n_process=nproc,
                    disable=["textcat", "tagger", "parser", "ner"],
                ),
            ),
            total=len(claims),
            desc="running spacy on claim.claim",
        )
    }
    return claim_docs


def get_claims(metadata_file):
    return Phase2Dataset.from_raw(metadata_file, setify=False).claims


########## Main ##########


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Main pipeline for the phase2 submission.")
    parser.add_argument("--metadata_file", type=str)
    parser.add_argument("--predictions_file", type=str)
    parser.add_argument("--claimant_model_file", type=str)
    parser.add_argument("--pretrained_model_name_or_path", type=str)
    parser.add_argument("--predict_batch_size", type=int, default=1)
    parser.add_argument("--nproc", type=int, default=1)
    parser_args = parser.parse_args()

    # read metadata
    log_title(_logger, "reading claims from {}".format(parser_args.metadata_file))
    claims = get_claims(parser_args.metadata_file)

    # generate claim spacy docs
    log_title(_logger, "generating claim spacy docs")
    claim_docs = generate_claim_docs(claims, nproc=parser_args.nproc)
    for claim in claims:
        claim.doc = claim_docs[claim.id]

    # fetch search api responses
    log_title(_logger, "fetching query responses")
    queries, responses = get_responses(claims, nproc=parser_args.nproc)

    null_responses = 0
    no_hits_responses = 0
    for i, claim in enumerate(claims):
        claim.query = queries[claim.id]
        claim.res = responses[claim.id]

        if i < 5:
            _logger.info("\nclaim: %s\nquery: %s\n", claim.logstr(), claim.query)

        if not claim.res:
            null_responses += 1
        elif not claim.res["hits"]["hits"]:
            no_hits_responses += 1

    _logger.info("  num_claims:         %d", len(claims))
    _logger.info("  null_responses:     %d", null_responses)
    _logger.info("  no_hits_responses:  %d", no_hits_responses)

    # select related articles
    log_title(_logger, "selecting related articles")
    articles_dict, related_articles_dict = get_related_articles(
        claims, keep_n_articles=2
    )
    for i, claim in enumerate(claims):
        claim.related_articles = related_articles_dict[claim.id]
        if i < 2:
            for article_id in claim.related_articles.values():
                _logger.info(
                    "\nclaim: %s\narticle: %s\n",
                    claim.logstr(),
                    articles_dict[article_id].logstr(),
                )

    # generate article spacy docs
    log_title(_logger, "generating articles spacy docs")
    article_docs = generate_article_docs(
        articles_dict.values(), nproc=parser_args.nproc
    )
    for article in articles_dict.values():
        article.doc = article_docs[article.id]

    # generate supporting article evidence
    log_title(_logger, "generating support")
    support_dict = generate_support(claims)
    for claim in claims:
        claim.support = support_dict[claim.id]

    # sequence classification predictions and explanations
    log_title(_logger, "generating sequence classification predictions")
    seq_clf_predictions, seq_clf_explanations = sequence_classification(
        claims=claims,
        pretrained_model_name_or_path=parser_args.pretrained_model_name_or_path,
        predict_batch_size=parser_args.predict_batch_size,
        nproc=parser_args.nproc,
    )
    for i, claim in enumerate(claims):
        if i > 5:
            break
        _logger.info(
            "\nclaim: %s\npred: %d\nexplanation: %s\n\n",
            claim.logstr(),
            seq_clf_predictions[claim.id],
            seq_clf_explanations[claim.id],
        )

    # claimant classification predictions and explanations
    log_title(_logger, "generating claimant predictions")
    claimant_predictions, claimant_explanations = claimant_classification(
        claims=claims, claimant_model_file=parser_args.claimant_model_file
    )
    for i, claim in enumerate(claims):
        if i > 5:
            break
        _logger.info(
            "\nclaim: %s\npred: %d\nexplanation: %s\n\n",
            claim.logstr(),
            claimant_predictions[claim.id],
            claimant_explanations[claim.id],
        )

    # compile final output
    log_title(_logger, "compiling output")
    output = compile_final_output(
        claims=claims,
        seq_clf_predictions=seq_clf_predictions,
        claimant_predictions=claimant_predictions,
    )
    _logger.info(json.dumps(dict(list(output.items())[:5]), indent=2))

    # write output to json
    log_title(_logger, "writing predictions to {}".format(parser_args.predictions_file))
    with open(parser_args.predictions_file, "w") as fo:
        json.dump(output, fo, indent=2)
    if not os.path.exists(parser_args.predictions_file):
        raise ValueError("predictions file was not created")

    # done :)
    log_title(_logger, "done")
