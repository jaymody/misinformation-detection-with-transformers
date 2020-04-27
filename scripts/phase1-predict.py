import json
import argparse
import collections

import numpy as np
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    InputExample,
)

from valerie.utils import get_logger
from valerie.datasets import BasicDataset

_logger = get_logger()


def load_examples(examples_file):
    _logger.info("... loading examples from %s ...", examples_file)
    with open(examples_file) as fi:
        examples = [InputExample(**example) for example in json.load(fi)]
    return examples


def generate_predictions(examples, pretrained_model_name_or_path, batch_size, ngpu, nproc):
    _logger.info("... loading config, tokenizer, and model from %s ...", pretrained_model_name_or_path)
    config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path, config=config)

    _logger.info("... loading dataset ...")
    predict_dataset = BasicDataset(examples, tokenizer, [0,1,2], nproc=nproc)
    predict_args = TrainingArguments(
        eval_batch_size=batch_size,
        n_gpu=ngpu,
        do_predict=True,
    )

    _logger.info("... loading trainer ...")
    trainer = Trainer(model=model, args=predict_args)

    _logger.info("... predicting ...")
    output = trainer.predict(predict_dataset)

    _logger.info("... averaging predictions across claim_id ...")
    assert len(examples) == len(output.predictions)
    probs = collections.defaultdict(list)
    for example, prob in zip(examples, output.predictions):
        probs[example.guid].append(prob)
    averaged_probs = {k: np.mean(v, axis=0) for k,v in probs.items()}
    predictions = {k: np.argmax(v) for k,v in averaged_probs.items()}

    return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generate phase1 predictions.")
    parser.add_argument("--examples_file", type=str)
    parser.add_argument("--predictions_file", type=str)
    parser.add_argument("--pretrained_model_name_or_path", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--ngpu", type=int)

    args = parser.parse_args()
    examples = load_examples(args.examples_file)
    predictions = generate_predictions(
        examples,
        args.pretrained_model_name_or_path,
        args.batch_size,
        args.ngpu,
        args.nproc
    )

    _logger.info("... saving predictions to %s ...", args.predictions_file)
    with open(args.predictions_file, 'w') as fo:
        for claim_id, pred in predictions.items():
            fo.write("%d,%d\n" % (claim_id, pred))

