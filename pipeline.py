import os
import json
import heapq
import random
import argparse
import warnings
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
from valerie.preprocessing import clean_text, extract_words_from_url

_logger = get_logger(use_tqdm_handler=False)

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
social_media_claimants = [
    "multiple websites",
    "various websites",
    "viral meme",
    "social media posts",
    "viral image",
    "facebook user",
    "chain email",
    "chain message",
    "instagram posts",
    "bloggers",
    "facebook posts",
    "tweets",
    "facebook post",
]

log_title(_logger, "loading spacy")
nlp = spacy.load("en_core_web_lg")

# ignore spacy doc similarity error for docs with empty vectors
warnings.filterwarnings("ignore", message=r"\[W008\]", category=UserWarning)


######################################
########## Helper Functions ##########
######################################


def _sequence_classification(
    examples, pretrained_model_name_or_path, predict_batch_size, nproc
):
    model = SequenceClassificationModel.from_pretrained(pretrained_model_name_or_path)

    predict_dataset = model.create_dataset(examples=examples, nproc=nproc)
    prediction_output = model.predict(predict_dataset, predict_batch_size)

    # returns an array of probs
    probabilities = prediction_output.predictions
    return probabilities if probabilities is not None else []


############################
########## Claims ##########
############################


def get_claims(metadata_file):
    return Phase2Dataset.from_raw(metadata_file, setify=False).claims


def generate_claim_docs_dict(claims):
    claim_docs_dict = {}
    for claim in tqdm(claims, desc="running spacy on claim.claim"):
        claim_docs_dict[claim.id] = nlp(
            claim.claim, disable=["textcat", "tagger", "parser", "ner"]
        )
    return claim_docs_dict


############################
########## Search ##########
############################


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
    # visited_url is probably uneeded but hopefully visited_content will prevent
    # exact duplicates (ie the same page but with http vs https, or different
    # args at the end of the url)
    visited_url = set()
    visited_content = set()
    output = []

    for hit in res["hits"]["hits"]:
        if hit["url"] in visited_url:
            continue

        article = Article.from_html(hit["url"], hit["content"], url=hit["url"])
        if (
            not article.content
            or len(article.content) < 32
            or article.content in visited_content
        ):
            continue

        assert article.url == article.id
        output.append({"score": hit["score"], "article": article, "url": hit["url"]})
        visited_url.add(article.url)
        visited_content.add(article.content)

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


def generate_text_a_dict(claims):
    def generate_text_a_text(claim):
        text = claim.claim
        text += " "
        text += claim.claimant if claim.claimant else "no claimant"
        text += " "
        text += claim.date.split()[0].split("T")[0] if claim.date else "no date"
        return clean_text(text)

    claim_text_a_dict = {}
    for claim in tqdm(claims, desc="running spacy on claim text_a"):
        doc_text = generate_text_a_text(claim)
        claim_text_a_dict[claim.id] = nlp(
            doc_text, disable=["textcat", "tagger", "parser", "ner"]
        )
    return claim_text_a_dict


##############################
########## Articles ##########
##############################


def get_articles_dict(claims):
    articles_dict = {}
    for claim in claims:
        if not claim.res:
            continue
        for hit in claim.res["hits"]["hits"]:
            article = hit["article"]
            articles_dict[article.id] = article
    return articles_dict


def get_hits_dict(claims, articles_dict):
    # we need to use articles_dict here because a claim can have
    hits_dict = {}
    for claim in claims:
        hits_dict[claim.id] = (
            [
                {
                    "url": hit["url"],
                    "article": articles_dict[hit["url"]],
                    "score": hit["score"],
                }
                for hit in claim.res["hits"]["hits"]
            ]
            if claim.res
            else []
        )
    return hits_dict


def generate_article_docs_dict(articles):
    def generate_article_doc_text(article):
        text = ""
        if article.source:
            text += article.source + ". "
        if article.title:
            text += article.title + ". "
        if article.url:
            url_words = extract_words_from_url(article.url)
            if url_words:
                text += " ".join(url_words) + ". "
        if article.content:
            text += article.content
        return clean_text(text)

    article_docs_dict = {}
    for article in tqdm(articles, desc="running spacy on articles"):
        doc_text = generate_article_doc_text(article)
        article_docs_dict[article.id] = nlp(
            doc_text, disable=["textcat", "tagger", "ner"]
        )

    return article_docs_dict


#############################
########## Support ##########
#############################


def generate_support_dict(claims, keep_top_n_sentences=32):
    # we use this for initialization instead of collection.defaultdict(dict)
    # because there is a possibilty that a claim has no hits in which case we
    # still want to register an entry for the claim (an empty on at that)
    support_dict = {claim.id: {} for claim in claims}

    for claim in tqdm(claims, desc="generating support"):
        for hit in claim.hits:
            article = hit["article"]

            support = []
            for sent in article.doc.sents:
                support.append(
                    {"text": sent.text, "score": claim.text_a.similarity(sent)}
                )
            support = heapq.nlargest(
                keep_top_n_sentences, support, key=lambda x: x["score"]
            )
            support_dict[claim.id][article.id] = support

    return support_dict


############################
########## Rerank ##########
############################


def generate_rerank_examples(claims):
    def generate_text_b(article, claim):
        text_b = clean_text(" ".join([s["text"] for s in claim.support[article.id]]))
        return text_b

    examples = []
    for claim in tqdm(claims, desc="generating rerank examples"):
        for hit in claim.hits:
            article = hit["article"]
            article.text_b = generate_text_b(article, claim)

            examples.append(
                SequenceClassificationExample(
                    guid=claim.id,
                    text_a=claim.text_a.text,
                    text_b=article.text_b,
                    art_id=article.id,
                )
            )
    return examples


def rerank_hits(claims, rerank_model_dir, predict_batch_size, keep_top_n, nproc):
    examples = generate_rerank_examples(claims)
    _logger.info(
        "first 5 rerank examples:\n%s",
        json.dumps([example.__dict__ for example in examples[:5]], indent=2),
    )

    probabilities = _sequence_classification(
        examples, rerank_model_dir, predict_batch_size=predict_batch_size, nproc=nproc,
    )

    if len(probabilities) != len(examples):
        raise ValueError(
            "len predictions ({}) != len examples ({})".format(
                len(probabilities), len(examples)
            )
        )

    # we use this for initialization instead of collection.defaultdict(list)
    # because there is a possibilty that a claim had a null responses when
    # queried, which means it has no rerank examples, in which case we still
    # want it in the below dict, but would be empty
    rerank_hits_dict = {claim.id: [] for claim in claims}
    for example, proba in tqdm(zip(examples, probabilities)):
        proba = float(proba[1])  # gets relatedness score of example
        rerank_hits_dict[example.guid].append(
            {"art_id": example.art_id, "score": proba}
        )

    for k, hits in rerank_hits_dict.items():
        top_n_hits = heapq.nlargest(keep_top_n, hits, key=lambda x: x["score"])
        rerank_hits_dict[k] = [
            {"art_id": x["art_id"], "score": x["score"]} for x in top_n_hits
        ]

    return rerank_hits_dict


#############################################
########## Sequence Classification ##########
#############################################


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


def sequence_classification(claims, fnc_model_dir, predict_batch_size, nproc):
    examples = generate_sequence_classification_examples(claims)
    _logger.info(
        "first 5 sequence classification examples:\n%s",
        json.dumps([example.__dict__ for example in examples[-5:]], indent=2),
    )

    probabilities = _sequence_classification(
        examples, fnc_model_dir, predict_batch_size=predict_batch_size, nproc=nproc,
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
                "The AI model detected some inconsitencies with the claim and "
                "the information in the articles. It's possible that the claim "
                "is biased and/or exaggerated to fit a certain narrative."
            )
        else:
            explanations[example.guid] = (
                "The AI model couldn't detect any patterns in the claim "
                "consitent with misinformation, clickbait, disinformation, or "
                "fake news. {}, the contents of the articles do not contradict "
                "the claim. This claim is likely true.".format(
                    random.choice(furthermore_syns)
                )
            )

    return predictions, explanations


#############################################
########## Claimant Classification ##########
#############################################


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
                "This is also substantiated by looking at the claimant's history "
                "of misinformation, having previously made {} (out of {}) "
                "false statements.".format(
                    claimant_model.model[claim.claimant.lower()]["false"],
                    claimant_model.model[claim.claimant.lower()]["total"],
                )
            )
        elif pred == 1:
            explanations[claim.id] = (
                "This is also supported by the fact that the claimant has a mixed "
                "record, having previously made {} (out of {}) partially "
                "correct/incorrect statements.".format(
                    claimant_model.model[claim.claimant.lower()]["partly"],
                    claimant_model.model[claim.claimant.lower()]["total"],
                )
            )
        else:
            explanations[claim.id] = (
                "This is further backed by the claimant's good "
                "track record, having previously made "
                "{} (out of {}) factual statements.".format(
                    claimant_model.model[claim.claimant.lower()]["true"],
                    claimant_model.model[claim.claimant.lower()]["total"],
                )
            )

    return predictions, explanations


##########################################
########## Compile Final Output ##########
##########################################


def compile_final_output(
    claims, articles_dict, seq_clf_predictions, claimant_predictions
):
    # sort articles by the relatedness of it's most relevant article
    claims = sorted(
        claims,
        key=lambda x: max([hit["score"] for hit in x.related_articles])
        if x.related_articles
        else 0,
        reverse=True,
    )

    output = {}
    for claim_iter, claim in enumerate(claims):
        # predicted label
        pred = seq_clf_predictions[claim.id]

        # related articles
        top_2_articles = claim.related_articles[:2]
        related_articles = {i + 1: x["art_id"] for i, x in enumerate(top_2_articles)}
        related_articles_inv = {v: k for k, v in related_articles.items()}

        # for an article to be given article sentences for it's explanation,
        # it needs to be in the top N% of claims sorted by the relatedness of
        # of it's most relevant article. this way, in theory, the articles that
        # are of higher quality (and are more likely to be related) will use support
        # while the "worse" articles will be given the attention, claimant, and
        # pattern related explanations that are more relevant (typically malicous
        # fake news and lesser talked about news falls in this category, which
        # is more likely to be applicable to the described explanations)
        support_articles = {}
        if claim_iter < round(len(claims) * 0.70):
            for art_num, hit in enumerate(top_2_articles):
                # don't use the second article if it's score is below 3.0
                if art_num < 1 or hit["score"] > 0.3:
                    support_articles[related_articles_inv[hit["art_id"]]] = hit[
                        "art_id"
                    ]

        relevant_support = []
        for art_num, art_id in support_articles.items():
            # this should be the case already, but just incase ...
            if art_id in claim.support:
                for sup in claim.support[art_id]:
                    relevant_support.append(
                        {**sup, "art_id": art_id, "art_num": art_num}
                    )
        relevant_support = heapq.nlargest(2, relevant_support, key=lambda x: x["score"])

        explanation = []
        first_art_id = None
        for sup in relevant_support:
            sup_text = sup["text"]
            if len(sup_text) > 340:
                sup_text = sup_text[:335] + " ..."

            article = articles_dict[sup["art_id"]]
            if len(explanation) == 0 and pred == 2:
                explanation.append(
                    "The claim was confirmed{} in the {} article, "
                    '"{}".'.format(
                        " by " + (article.source if article.source else ""),
                        number2place[sup[art_num]],
                        sup_text,
                    )
                )
            elif len(explanation) == 0:
                explanation.append(
                    "The claim is {}, as explained in the {} "
                    'article, which wrote "{}".'.format(
                        id2label[pred], number2place[sup[art_num]], sup_text,
                    )
                )
                first_art_id = sup["art_id"]
            elif sup["art_id"] == first_art_id and first_art_id is not None:
                explanation.append(
                    '{}, the article goes on to say "{}".'.format(
                        random.choice(furthermore_syns), sup_text,
                    )
                )
            else:
                explanation.append(
                    "This conclusion can also be drawn from article {}{}, "
                    'stating "{}".'.format(
                        sup[art_num],
                        " by " + article.source if article.source else "",
                        sup_text[:400],
                    )
                )

        # social media related explanations
        if not explanation:
            if claim.claimant in social_media_claimants and pred == 0:
                explanation.append(
                    "The claim contains patterns consitent with malicious and/or "
                    "clickbait fake news designed to spread fear. "
                    "This claim falls in line with other "
                    "social media based misinformation, that is spread through "
                    "the claims ability to outrage the reader."
                )
            elif claim.claimant in social_media_claimants and pred == 1:
                explanation.append(
                    "The AI model detected patterns consitent with clickbait and/or "
                    "misrepresentation. There may be some truth to the claim, "
                    "however it does't provide the whole picture and contains "
                    "biases."
                )
            else:
                explanation.append(seq_clf_explanations[claim.id])

        if claimant_predictions[claim.id] == pred:
            explanation.append(claimant_explanations[claim.id])

        explanation = [e for e in explanation if e]
        explanation = " ".join(explanation)
        if len(explanation) > 995:
            explanation = explanation[:995] + " ..."
        explanation = explanation[:999]

        # test outputs are correct
        assert isinstance(pred, int)
        assert pred in label_set

        assert isinstance(related_articles, dict)
        assert len(related_articles) >= 0 and len(related_articles) <= 2
        for k, v in related_articles.items():
            assert k in {1, 2}
            assert isinstance(v, str)

        assert isinstance(explanation, str)
        assert len(explanation) < 1000

        # final predictions
        output[claim.id] = {
            "label": pred,
            "related_articles": related_articles,
            "explanation": explanation,
        }
    return output


##########################
########## Main ##########
##########################


if __name__ == "__main__":
    ###########
    ### cli ###
    ###########
    parser = argparse.ArgumentParser("Main pipeline for the phase2 submission.")
    parser.add_argument("--metadata_file", type=str)
    parser.add_argument("--predictions_file", type=str)
    parser.add_argument("--claimant_model_file", type=str)
    parser.add_argument("--fnc_model_dir", type=str)
    parser.add_argument("--rerank_model_dir", type=str)
    parser.add_argument("--predict_batch_size", type=int, default=1)
    parser.add_argument("--nproc", type=int, default=1)
    parser_args = parser.parse_args()

    #####################
    ### read metadata ###
    #####################
    log_title(_logger, "reading claims from {}".format(parser_args.metadata_file))
    claims = get_claims(metadata_file=parser_args.metadata_file)
    log_msg = ""
    for i, claim in enumerate(claims):
        if i >= 5:
            break
        log_msg += "\n{}\n".format(claim.logstr())
    _logger.info("first 5 claims:\n%s", log_msg)

    ######################
    ### process claims ###
    ######################
    log_title(_logger, "process claims")
    claim_docs_dict = generate_claim_docs_dict(claims=claims)
    log_msg = ""
    for i, claim in enumerate(claims):
        claim.doc = claim_docs_dict[claim.id]
        if i < 5:
            log_msg += "\nclaim_id = {}\n{}\n".format(claim.id, claim.doc.text)
    _logger.info("first 5 claim doc texts:\n%s", log_msg)

    claim_text_a_dict = generate_text_a_dict(claims=claims)
    log_msg = ""
    for i, claim in enumerate(claims):
        claim.text_a = claim_text_a_dict[claim.id]
        if i < 5:
            log_msg += "\nclaim_id = {}\n{}\n".format(claim.id, claim.text_a)
    _logger.info("first 5 claim text_a:\n%s", log_msg)

    #######################
    ### fetch responses ###
    #######################
    log_title(_logger, "fetching query responses")
    queries, responses = get_responses(claims=claims, nproc=parser_args.nproc)

    log_msg = ""
    null_responses = 0
    for i, claim in enumerate(claims):
        claim.query = queries[claim.id]
        claim.res = responses[claim.id]

        if i < 5:
            log_msg += "\nclaim_id = {}\n{}\n".format(claim.id, claim.query)

        if not claim.res:
            null_responses += 1

    _logger.info("  num_claims:         %d", len(claims))
    _logger.info("  null_responses:     %d", null_responses)
    _logger.info("first 5 generated queries:\n%s", log_msg)

    ########################
    ### process articles ###
    ########################
    log_title(_logger, "process related articles")
    articles_dict = get_articles_dict(claims=claims)
    _logger.info("  num articles in articles_dict:  %d", len(articles_dict))

    hits_dict = get_hits_dict(claims=claims, articles_dict=articles_dict)
    log_msg = ""
    num_total_hits = 0
    no_hits_claims = 0
    for i, claim in enumerate(claims):
        claim.hits = hits_dict[claim.id]
        num_total_hits += len(claim.hits)
        if i < 5:
            log_msg += "\nclaim_id = {}\n{}\n".format(
                claim.id,
                json.dumps(
                    [
                        {"score": hit["score"], "url": hit["url"]}
                        for hit in claim.hits[:5]
                    ],
                    indent=2,
                ),
            )
        if not claim.hits:
            no_hits_claims += 1
    _logger.info("  num_total_hits:     %d", num_total_hits)
    _logger.info("  no_hits_claims:     %d", no_hits_claims)
    _logger.info("first 5 claim hits (resticted to top 5 hits per claim):\n%s", log_msg)

    article_docs_dict = generate_article_docs_dict(articles=articles_dict.values())
    for article in articles_dict.values():
        article.doc = article_docs_dict[article.id]

    ########################
    ### generate support ###
    ########################
    log_title(_logger, "generating support")
    support_dict = generate_support_dict(claims=claims, keep_top_n_sentences=32)
    log_msg = ""
    for i, claim in enumerate(claims):
        claim.support = support_dict[claim.id]
        if i < 5:
            log_msg += "\nclaim_id = {}\n{}\n".format(
                claim.id,
                json.dumps(
                    {k: v[:3] for k, v in list(claim.support.items())[:3]}, indent=2
                ),
            )
    _logger.info(
        "first 5 claim support (top 3 articles, top 3 sentences):\n%s", log_msg
    )

    ##############
    ### rerank ###
    ##############
    log_title(_logger, "rerank and select top 2 articles")
    rerank_hits_dict = rerank_hits(
        claims=claims,
        rerank_model_dir=parser_args.rerank_model_dir,
        predict_batch_size=parser_args.predict_batch_size,
        keep_top_n=2,  # make sure we only keep the top two results
        nproc=parser_args.nproc,
    )
    log_msg = ""
    for i, claim in enumerate(claims):
        claim.related_articles = rerank_hits_dict[claim.id]
        if i < 5:
            for hit in claim.related_articles:
                log_msg += "\nclaim_id = {}\narticle score = {:.3f}\n{}\n".format(
                    claim.id, hit["score"], articles_dict[hit["art_id"]].logstr(),
                )
    _logger.info("first 5 claims chosen reranked articles:\n%s", log_msg)

    ###############################
    ### sequence classification ###
    ###############################
    log_title(_logger, "generating sequence classification predictions")
    seq_clf_predictions, seq_clf_explanations = sequence_classification(
        claims=claims,
        fnc_model_dir=parser_args.fnc_model_dir,
        predict_batch_size=parser_args.predict_batch_size,
        nproc=parser_args.nproc,
    )
    log_msg = ""
    for i, claim in enumerate(claims):
        if i > 5:
            break
        log_msg += "\nclaim_id = {}\npred: {}\nexplanation: {}\n".format(
            claim.id,
            str(seq_clf_predictions[claim.id]),
            seq_clf_explanations[claim.id],
        )
    _logger.info("first 5 sequence classification output results:\n%s", log_msg)

    ###############################
    ### claimant classification ###
    ###############################
    log_title(_logger, "generating claimant predictions")
    claimant_predictions, claimant_explanations = claimant_classification(
        claims=claims, claimant_model_file=parser_args.claimant_model_file
    )
    log_msg = ""
    for i, claim in enumerate(claims):
        if i > 5:
            break
        log_msg += "\nclaim_id = {}\npred: {}\nexplanation: {}\n".format(
            claim.id,
            str(claimant_predictions[claim.id]),
            claimant_explanations[claim.id],
        )
    _logger.info("first 5 claimant classification output results:\n%s", log_msg)

    ############################
    ### compile final output ###
    ############################
    log_title(_logger, "compiling output")
    output = compile_final_output(
        claims=claims,
        articles_dict=articles_dict,
        seq_clf_predictions=seq_clf_predictions,
        claimant_predictions=claimant_predictions,
    )
    _logger.info(
        "first 5 output entries:\n%s",
        json.dumps(dict(list(output.items())[:5]), indent=2),
    )

    #####################################
    ### write output predictions file ###
    #####################################
    log_title(_logger, "writing predictions to {}".format(parser_args.predictions_file))
    with open(parser_args.predictions_file, "w") as fo:
        json.dump(output, fo, indent=2)
    if not os.path.exists(parser_args.predictions_file):
        raise ValueError("predictions file was not created")

    # done :)
    log_title(_logger, "done :)")
