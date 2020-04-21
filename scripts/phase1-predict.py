import os
import json
import glob
import argparse
import collections
import multiprocessing

import numpy as np
from tqdm import tqdm

from valerie.utils import get_logger
from valerie.modeling import predict

_logger = get_logger()


def generate_predictions(examples_file, predictions_file, pretrained_model_name_or_path, use_cuda, cuda_device, batch_size, ngpu):
    _logger.info("... loading examples from {} ...".format(examples_file))
    with open(examples_file, 'r') as fi:
        examples = json.load(fi)

    _logger.info("... computing predictions ...")
    output = predict(
        examples,
        pretrained_model_name_or_path,
        use_cuda=use_cuda,
        cuda_device=cuda_device,
        batch_size=batch_size
    )

    _logger.info("... averaging predictions across claim_id ...")
    probs = collections.defaultdict(list)
    for example, prob in zip(examples, output[1]):
        probs[example.guid].append(prob)
    averaged_probs = {k: np.mean(v, axis=0) for k,v in probs.items()}
    predictions = {k: np.argmax(v) for k,v in averaged_probs.items()}

    _logger.info("... saving predictions to {} ...".format(predictions_file))
    with open(predictions_file, 'w') as fo:
        for claim_id, pred in predictions.items():
            fo.write("%d,%d\n" % (claim_id, pred))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generate phase1 predictions.")
    parser.add_argument("--examples_file", type=str)
    parser.add_argument("--predictions_file", type=str)
    parser.add_argument("--pretrained_model_name_or_path", type=str)
    parser.add_argument("--use_cuda", type=bool)
    parser.add_argument("--cuda_device", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--ngpu", type=int)

    kwargs = parser.parse_args()
    generate_predictions(**kwargs.__dict__)

