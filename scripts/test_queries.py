import os
import time
import json
import itertools

import nltk
import spacy
from tqdm import tqdm

from valerie import search
from valerie.data import load_claims
from valerie.utils import get_logger
from valerie.scoring import validate_predictions_phase2, compute_score_phase2
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


def load_responses(filepath):
    with open(filepath) as fi:
        return json.load(fi)


def save_responses(responses, filepath):
    with open(filepath, "w") as fo:
        json.dump(responses, fo, indent=2)


def all_query(claim, do_ner=True, do_claimant=True, do_stopword=True, from_idx=0):
    dis = ["textcat", "tagger", "parser"]
    if not do_ner:
        dis += ["ner"]
    claim_doc = nlp(claim.claim, disable=dis)

    # generate query string
    query_words = [
        token.text for token in claim_doc if not do_stopword or not token.is_stop
    ]
    if do_claimant and claim.claimant:
        query_words.insert(0, claim.claimant)
    if do_ner and len(query_words) < 8:
        query_words += [ent.text for ent in claim_doc.ents]
    query = clean_text(" ".join(query_words), remove_punctuation=True)

    # get response
    res = search.query(query, from_idx=from_idx)
    return claim, query, res


if __name__ == "__main__":
    claims = load_claims("data/phase2/all_data/claims.json")
    claims_list = list(claims.values())

    ##### Run All Perumatationns of Params #####
    params = {
        "do_ner": [False, True],
        "do_claimant": [True],
        "do_stopword": [True],
        "from_idx": [0],
    }
    run_configs = []
    keys = list(params)
    for values in itertools.product(*map(params.get, keys)):
        run_configs.append(dict(zip(keys, values)))

    save_dir = "data/phase2/queries/500"
    run_claims = claims_list[4000:4500]
    assert "nlp" in globals()

    for i, run_config in enumerate(run_configs):
        print(f"run {i}".center(50, "-"))
        print(json.dumps(run_config, indent=2))
        time.sleep(1)
        with open(os.path.join(save_dir, f"{i}_run_config.json"), "w") as fo:
            json.dump(run_config, fo, indent=2)

        responses = {}
        for claim in tqdm(run_claims):
            claim, query, res = all_query(claim, **run_config)
            responses[claim.id] = {"id": claim.id, "query": query, "res": res}

        save_responses(responses, os.path.join(save_dir, f"{i}_run_responses.json"))

        with open(os.path.join(save_dir, f"{i}_run_score.json"), "w") as fo:
            scores = compute_query_score(responses, claims)
            print(json.dumps(scores, indent=2))
            json.dump(scores, fo, indent=2)

        print()

    ##### Vanilla Run #####
    # responses = {}
    # start = time.time()
    # for k, v in tqdm(claims.items()):
    #     responses[k] = search.query(v.claim)

    # _logger.info("time took: {}".format(time.time() - start))
    # save_responses(responses, "data/phase2/vanilla_responses.json")
