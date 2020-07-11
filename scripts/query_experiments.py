import os
import time
import json
import pickle
import random
import inspect
import argparse
import itertools

import spacy
from tqdm.auto import tqdm

from valerie import search
from valerie.datasets import Phase2Dataset
from valerie.utils import get_logger
from valerie.scoring import validate_predictions_phase2, _compute_score_phase2
from valerie.preprocessing import clean_text

from query import compute_responses_score

nlp = spacy.load("en_core_web_lg")


def run_config_combinations(query_params):
    run_configs = []
    keys = list(query_params)
    for values in itertools.product(*map(query_params.get, keys)):
        run_configs.append(dict(zip(keys, values)))
    return run_configs


def query_func(
    claim,
    do_date,
    do_ner,
    do_claimant,
    do_stopword,
    do_lemma,
    add_beginning,
    from_idx=0,
):
    # get spacy nlp data for claim
    dis = ["textcat", "tagger", "parser"]
    if not do_ner:
        dis += ["ner"]
    claim_doc = nlp(claim.claim, disable=dis)

    # stopword removal
    query_tokens = [
        token for token in claim_doc if not do_stopword or not token.is_stop
    ]

    # lemmatization
    if do_lemma:
        query_words = [token.lemma_ for token in query_tokens]
    else:
        query_words = [token.text for token in query_tokens]

    # clean text
    query = clean_text(
        " ".join(
            [
                t
                for t in query_words
                if t and not len(clean_text(t, remove_punctuation=True)) == 0
            ]
        )
    )

    # ner
    if do_ner:
        ner_str = clean_text(
            " ".join(set([ent.text for ent in claim_doc.ents if ent.text not in query]))
        )
        if ner_str:
            query += " " + ner_str

    # date
    if do_date and claim.date:
        if add_beginning:
            query = claim.date.split(" ")[0].split("T")[0] + " " + query
        else:
            query += " " + claim.date.split(" ")[0]

    # claimant
    if do_claimant and claim.claimant:
        if add_beginning:
            query = claim.claimant + " " + query
        else:
            query += " " + claim.claimant

    # get server query response
    res = search.query(query, from_idx=from_idx)
    return claim, query, res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--metadata_file", type=str)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument(
        "--query_params",
        type=dict,
        default={
            "do_date": [True],  # it's been proven that adding date helps
            "do_ner": [False],  # adding ner doesn't seem to make a difference
            "do_claimant": [True],  # it's been proven that adding claimant helps
            "do_stopword": [True],  # it's been proven stopword removal helps
            "do_lemma": [False, True],
            "add_beginning": [False],  # doesn't make a difference
            "from_idx": [0],
        },
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir)
    with open(os.path.join(args.output_dir, "args.json"), "w") as fo:
        json.dump(args.__dict__, fo, indent=2)
    with open(os.path.join(args.output_dir, "query_func.txt"), "w") as fo:
        fo.write(inspect.getsource(query_func))

    _logger = get_logger(os.path.join(args.output_dir, "log.txt"))
    claims = Phase2Dataset.from_raw(args.metadata_file).claims
    random.seed(args.seed)
    n_samples = args.n_samples if args.n_samples is not None else len(claims)
    run_claims = random.sample(claims, n_samples)
    run_configs = run_config_combinations(args.query_params)

    output = {}
    for i, run_config in enumerate(run_configs):
        _logger.info(f"run {i}".center(50, "-"))
        _logger.info("config: %s", json.dumps(run_config, indent=2))
        time.sleep(1)  # sleep so logging isn't interupted

        responses = []
        for claim in tqdm(run_claims):
            claim, query, res = query_func(claim, **run_config)

            # discard response html data since it will overload ram
            if res:
                for hit in res["hits"]["hits"]:
                    hit["content"] = ""

            responses.append({"claim": claim, "query": query, "res": res})

        scores = compute_responses_score(responses)
        _logger.info("scores: %s", json.dumps(scores, indent=2))

        output[i] = {"config": run_config, "scores": scores}

    with open(os.path.join(args.output_dir, "results.json"), "wb") as fo:
        pickle.dump(output, fo)
