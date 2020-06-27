import os
import json
import argparse

import numpy as np
from tqdm.auto import tqdm
from sklearn.metrics import classification_report

from valerie.utils import get_logger
from valerie.modeling import SequenceClassificationModel


_logger = get_logger()


def get_args_dicts(output_dir):
    if not os.path.exists(output_dir):
        raise ValueError(f"output dir does not exists: {output_dir}")

    args_list = [
        "data_args",
        "config_args",
        "tokenizer_args",
        "model_args",
        "training_args",
    ]

    args_dict = {}
    for arg in args_list:
        with open(os.path.join(output_dir, arg + ".json")) as fi:
            args_dict[arg] = json.load(fi)

    return args_dict


def get_report(examples, output):
    _labels = []
    _preds = []
    for example, prob in zip(examples, output.predictions):
        _labels.append(example.label)
        _preds.append(np.argmax(prob))
    return classification_report(_labels, _preds)


def train(
    output_dir,
    pretrained_model_name_or_path,
    train_examples_file,
    test_examples_file=None,
    trial_examples_file=None,
    nproc=1,
):
    """Train sequence classifier."""
    args_dicts = get_args_dicts(output_dir)

    train_examples = SequenceClassificationModel.load_examples(train_examples_file)
    test_examples = None
    if test_examples_file is not None:
        test_examples = SequenceClassificationModel.load_examples(test_examples_file)

    model, _, test_dataset = SequenceClassificationModel.train_from_pretrained(
        output_dir=output_dir,
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        train_examples=train_examples,
        test_examples=test_examples,
        nproc=nproc,
        **args_dicts,
    )

    test_output = model.predict(test_dataset, predict_batch_size=1)
    test_report = get_report(test_examples, test_output)
    _logger.info("test data classification report:\n\n%s\n" % test_report)
    with open(os.path.join(output_dir, "eval_report_test_data.txt"), "w") as fo:
        fo.write(test_report)

    if trial_examples_file:
        trial_examples = SequenceClassificationModel.load_examples(trial_examples_file)
        trial_dataset = model.create_dataset(trial_examples)
        trial_output = model.predict(trial_dataset, predict_batch_size=1)
        trial_report = get_report(trial_examples, trial_output)
        _logger.info("trial data classification report:\n\n%s\n" % trial_report)
        with open(os.path.join(output_dir, "eval_report_trial_data.txt"), "w") as fo:
            fo.write(trial_report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train sequence classifier.")
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--pretrained_model_name_or_path", type=str)
    parser.add_argument("--train_examples_file", type=str)
    parser.add_argument("--test_examples_file", type=str, default=None)
    parser.add_argument("--trial_examples_file", type=str, default=None)
    parser.add_argument("--nproc", type=int, default=1)
    args = parser.parse_args()

    train(**args.__dict__)
