"""Predict given preprocessed data and trained model."""
import json
import logging
import collections

import numpy as np
from simpletransformers.model import TransformerModel

_logger = logging.getLogger(__name__)


def load_params(params_fpath, nproc=1, ngpu=0, eval_batch_size=None):
    with open(params_fpath, 'r') as fi:
        params = json.load(fi)
    params['process_count'] = nproc
    params['n_gpu'] = ngpu
    if eval_batch_size:
        params['eval_batch_size'] = eval_batch_size
    return params


def load_model(model_dir, model_args):
    _logger.info("... loading model ...")
    return TransformerModel('roberta', model_dir, num_labels=3, args=model_args)


def predict_proba(examples_text, claim_ids, model):
    _logger.info("... predicting ...")
    model_outputs = model.predict(examples_text)

    probs = collections.defaultdict(list)
    for claim_id, prob in zip(claim_ids, model_outputs[:][1]):
        probs[claim_id].append(prob)

    averaged_probs = {k: np.mean(v, axis=0) for k,v in probs.items()}
    return averaged_probs

def predict(examples_text, claim_ids, model):
    probs = predict_proba(examples_text, claim_ids, model)
    return {k: np.argmax(v) for k,v in probs.items()}


def main(data_path, predictions_fpath, model_dir, params_fpath, nproc, ngpu):
    ## Load Model
    model_args = load_params(params_fpath, nproc=nproc, ngpu=ngpu)
    model = load_model(model_dir, model_args)

    ## Predict
    df = pd.read_csv(data_path)
    predictions = predict(df["text"], df["id"], model)

    ## Write Predictions
    with open(predictions_fpath, 'w', encoding="utf-8") as fo:
        for claim_id, prediction in predictions.items():
            fo.write("%d,%d\n" % (claim_id, prediction))
    _logger.info("... wrote predictions to {} ...".format(predictions_fpath))


if __name__ == "__main__":
    # Main Imports
    import argparse
    import pandas as pd
    from preprocess import preprocess

    # CLI
    parser = argparse.ArgumentParser("Predict true, partly true, or false, for fake news data.")
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--predictions_fpath", type=str)
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--params_fpath", type=str)
    parser.add_argument("--nproc", type=int)
    parser.add_argument("--ngpu", type=int)

    args = parser.parse_args()
    main(**args.__dict__)
