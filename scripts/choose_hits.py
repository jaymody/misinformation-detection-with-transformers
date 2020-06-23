import json

import spacy
import numpy as np
from tqdm import tqdm

from valerie.data import Article
from valerie.utils import get_logger
from valerie.scoring import validate_predictions_phase2, compute_score_phase2
from valerie.modeling import SourceModel
from valerie.preprocessing import extract_words_from_url

_logger = get_logger()

nlp = spacy.load("en_core_web_lg")
source_model = SourceModel.from_pretrained("models/source_model.json")


def print_header(result):
    print("---- Claim ----")
    p_json = {
        k: v
        for k, v in result["claim"].to_dict().items()
        if k not in ["date", "support", "explanation"]
    }
    print(json.dumps(p_json, indent=2))
    print()

    print("---- Query ----")
    print("query:", result["query"])
    print()


def print_hits(result, n_hits=10):
    print(f"result {result['claim'].id}".center(50, "-"))

    print("---- Score ----")
    print(
        "query_hits:\t",
        sum(
            1
            for h in result["hits"]
            if h["url"] in result["claim"].related_articles.values()
        ),
    )
    print(
        "top_hits:\t",
        sum(
            1
            for h in result["hits"][:2]
            if h["url"] in result["claim"].related_articles.values()
        ),
    )
    print()

    print("---- Hits ----")
    p_json = [
        {k: v for k, v in h.items() if k not in ["article", "url"]}
        for h in result["hits"][:n_hits]
    ]
    print(json.dumps(p_json, indent=2))
    print()


# {
#     "claim": claim.claim,
#     "max_score": res["res"]["hits"]["max_score"],
#     "query": res["query"],
#     "hits": [
#         {
#             "id": article.id,
#             "scores": {},
#             "article": article,
#             "title": article.title,
#             "url": article.url,
#             "source": article.source,
#             "url_words": " ".join(url_words),
#         },
#     ],
# }


def compute_score_results(results):
    predictions = {}
    perfect_predictions = {}
    labels = {}

    for result in results.values():
        claim = result["claim"]
        labels[claim.id] = claim.to_dict()
        predictions[claim.id] = {
            "label": claim.label,
            "related_articles": [hit["url"] for hit in result["hits"][:2]],
            "explanation": "",
        }
        perfect_predictions[claim.id] = {
            "label": claim.label,
            "related_articles": [
                hit["url"]
                for hit in result["hits"]
                if hit["url"] in claim.related_articles.values()
            ][:2],
            "explanation": "",
        }

    validate_predictions_phase2(predictions)
    score = compute_score_phase2(labels, predictions)
    validate_predictions_phase2(perfect_predictions)
    perfect_score = compute_score_phase2(labels, perfect_predictions)
    return {
        "perfect_score": perfect_score["score"],
        "perfect_error": perfect_score["error"],
        "score": score["score"],
        "error": score["error"],
    }


def reweighted_score(hit, reweight_fn_dict=None):
    foos = (
        reweight_fn_dict
        if reweight_fn_dict
        else {
            "api": lambda x: 0.7 * x,
            "source": lambda x: x / 4 + 0.25,
            "w2v_url": lambda x: 0.6 * x,
            "w2v_content": lambda x: 0.8 * x,
            "w2v_title": lambda x: 0.6 * x,
            "claimant_url": lambda x: x / 2,
            "claimant_content": lambda x: x / 2,
            "claimant_title": lambda x: x / 2,
            "ner_content": lambda x: x / 2,
            "ner_url": lambda x: x / 2,
            "ner_title": lambda x: x / 2,
        }
    )
    return sum(foos[k](v) for k, v in hit["scores"].items())


def choose_hits(claim, res, n_hits=None):
    if not res["res"]["hits"]["hits"]:
        return None

    claim_doc = nlp(claim.claim, disable=["textcat", "tagger", "parser"])

    output = []
    for hit in res["res"]["hits"]["hits"][:n_hits]:
        article = Article.from_dict(hit["article"])

        url_words = extract_words_from_url(article.url)

        result = {
            "id": article.id,
            "scores": {
                "api": hit["score"] / res["res"]["hits"]["max_score"],
                "source": source_model.predict(article),
            },
            "article": article,
            "title": article.title,
            "url": article.url,
            "source": article.source,
            "url_words": " ".join(url_words),
        }

        if url_words and len(url_words) > 3:
            url_doc = nlp(
                " ".join(url_words), disable=["textcat", "tagger", "parser", "ner"]
            )
            result["scores"]["w2v_url"] = claim_doc.similarity(url_doc)
            result["scores"]["claimant_url"] = (
                1.0 if claim.claimant and claim.claimant in " ".join(url_words) else 0.0
            )
            for ent in claim_doc.ents:
                result["scores"]["ner_url"] = 0.0
                if ent.text in url_doc.text or ent.text.lower() in url_doc.text:
                    result["scores"]["ner_url"] = 1.0
                    break
                else:
                    for token in nlp(ent.text):
                        if token.text in url_doc.text:
                            result["scores"]["ner_url"] = 0.5
                            break

        if article.content and len(article.content) > 32:
            content_doc = nlp(
                article.content, disable=["textcat", "tagger", "parser", "ner"]
            )
            result["scores"]["w2v_content"] = claim_doc.similarity(content_doc)
            result["scores"]["claimant_content"] = (
                1.0 if claim.claimant and claim.claimant in content_doc.text else 0.0
            )
            for ent in claim_doc.ents:
                result["scores"]["ner_content"] = 0.0
                if ent.text in content_doc.text or ent.text.lower() in content_doc.text:
                    result["scores"]["ner_content"] = 1.0
                    break
                else:
                    for token in nlp(ent.text):
                        if token.text in content_doc.text:
                            result["scores"]["ner_content"] = 0.5
                            break

        if article.title and len(article.title) > 8:
            title_doc = nlp(
                article.title, disable=["textcat", "tagger", "parser", "ner"]
            )
            result["scores"]["w2v_title"] = claim_doc.similarity(title_doc)
            result["scores"]["claimant_title"] = (
                1.0 if claim.claimant and claim.claimant in title_doc.text else 0.0
            )
            for ent in claim_doc.ents:
                result["scores"]["ner_title"] = 0.0
                if ent.text in title_doc.text or ent.text.lower() in title_doc.text:
                    result["scores"]["ner_title"] = 1.0
                    break
                else:
                    for token in nlp(ent.text):
                        if token.text in title_doc.text:
                            result["scores"]["ner_title"] = 0.5
                            break

        # take average of existing scores
        scr = np.mean(list(result["scores"].values()))
        result["_score"] = scr
        result["score"] = scr

        # add scores not found for feature completeness
        possible_scores = [
            "api",
            "source",
            "w2v_url",
            "w2v_content",
            "w2v_title",
            "claimant_url",
            "claimant_content",
            "claimant_title",
            "ner_content",
            "ner_url",
            "ner_title",
        ]
        for p_score in possible_scores:
            if p_score not in result["scores"]:
                result["scores"][p_score] = None

        # add result to output
        output.append(result)

    output = sorted(output, key=lambda x: x["score"], reverse=True)
    return output
