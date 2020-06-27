import os
import copy
import json
import argparse

import numpy as np
from tqdm.auto import tqdm
from sklearn.metrics import classification_report

from valerie.utils import get_logger
from valerie.data import load_claims
from valerie.modeling import SequenceClassificationModel

_logger = get_logger()

experiments = [
    {"pretrained_model_name_or_path": "bert-base-cased", "training_args": {}},
    {
        "pretrained_model_name_or_path": "bert-base-cased",
        "training_args": {"max_grad_norm": 0.5},
    },
    {
        "pretrained_model_name_or_path": "bert-base-cased",
        "training_args": {"max_grad_norm": 2.0},
    },
    {
        "pretrained_model_name_or_path": "bert-base-cased",
        "training_args": {"adam_epsilon": 1e-8},
    },
    {
        "pretrained_model_name_or_path": "bert-base-cased",
        "training_args": {"adam_epsilon": 1e-4},
    },
    {
        "pretrained_model_name_or_path": "bert-base-cased",
        "training_args": {"learning_rate": 2e-3},
    },
    {
        "pretrained_model_name_or_path": "bert-base-cased",
        "training_args": {"learning_rate": 2e-4},
    },
    {
        "pretrained_model_name_or_path": "bert-base-cased",
        "training_args": {"learning_rate": 2e-5},
    },
    {
        "pretrained_model_name_or_path": "bert-base-cased",
        "training_args": {"learning_rate": 5e-6},
    },
    {
        "pretrained_model_name_or_path": "bert-base-cased",
        "training_args": {"learning_rate": 2e-6},
    },
    {
        "pretrained_model_name_or_path": "bert-base-cased",
        "training_args": {"warmup_steps": 500},
    },
    {
        "pretrained_model_name_or_path": "bert-base-cased",
        "training_args": {
            "per_device_train_batch_size": 8,
            "per_device_eval_batch_size": 8,
            "gradient_accumulation_steps": 4,
            "warmup_steps": 200,
            "logging_steps": 50,
        },
    },
    {
        "pretrained_model_name_or_path": "bert-base-cased",
        "training_args": {
            "per_device_train_batch_size": 8,
            "per_device_eval_batch_size": 8,
            "warmup_steps": 200,
            "logging_steps": 50,
        },
    },
    {
        "pretrained_model_name_or_path": "bert-base-cased",
        "training_args": {
            "per_device_train_batch_size": 16,
            "per_device_eval_batch_size": 16,
            "warmup_steps": 100,
            "logging_steps": 25,
        },
    },
    {
        "pretrained_model_name_or_path": "bert-base-cased",
        "training_args": {
            "per_device_train_batch_size": 64,
            "per_device_eval_batch_size": 64,
            "warmup_steps": 400,
            "logging_steps": 100,
        },
    },
]


def get_report(examples, output):
    _labels = []
    _preds = []
    for example, prob in zip(examples, output.predictions):
        _labels.append(example.label)
        _preds.append(np.argmax(prob))
    return classification_report(_labels, _preds)


if __name__ == "__main__":
    examples_dir = "models/phase2/single-claim-claimant-date"
    base_dir = os.path.join(examples_dir, "bert-base-cased-search")

    train_examples_file = os.path.join(examples_dir, "train_examples_combined.json")
    test_examples_file = os.path.join(examples_dir, "test_examples.json")
    trial_examples_file = os.path.join(examples_dir, "trial_examples.json")

    train_claims = load_claims("data/combined/phase1-phase2/claims.json")
    test_claims = load_claims("data/phase2/test-data/claims.json")
    trial_claims = load_claims("data/phase2/trial-data/claims.json")

    train_examples = SequenceClassificationModel.load_examples(train_examples_file)
    test_examples = SequenceClassificationModel.load_examples(test_examples_file)
    trial_examples = SequenceClassificationModel.load_examples(trial_examples_file)

    default_base_params = {
        "data_args": {
            "train_examples_file": train_examples_file,
            "test_examples_file": test_examples_file,
            "trial_examples_file": trial_examples_file,
        },
        "training_args": {
            "evaluate_during_training": True,
            "per_device_train_batch_size": 32,
            "per_device_eval_batch_size": 32,
            "gradient_accumulation_steps": 1,
            "learning_rate": 5e-5,
            "weight_decay": 0.00,
            "adam_epsilon": 1e-6,
            "max_grad_norm": 1.0,
            "num_train_epochs": 16,
            "warmup_steps": 50,
            "logging_first_step": False,
            "logging_steps": 15,
            "save_steps": 1e9,
            "save_total_limit": 1,
            "seed": 42,
        },
        "config_args": {
            "num_labels": 3,
            "id2label": {"0": "false", "1": "partly", "2": "true"},
            "label2id": {"false": 0, "partly": 1, "true": 2,},
        },
        "tokenizer_args": {"model_max_length": 128},
        "model_args": {},
    }

    default_large_params = {
        "data_args": {
            "train_examples_file": train_examples_file,
            "test_examples_file": test_examples_file,
            "trial_examples_file": trial_examples_file,
        },
        "training_args": {
            "evaluate_during_training": True,
            "per_device_train_batch_size": 16,
            "per_device_eval_batch_size": 16,
            "gradient_accumulation_steps": 1,
            "learning_rate": 5e-5,
            "weight_decay": 0.00,
            "adam_epsilon": 1e-6,
            "max_grad_norm": 1.0,
            "num_train_epochs": 8,
            "warmup_steps": 100,
            "logging_first_step": False,
            "logging_steps": 25,
            "save_steps": 1e9,
            "save_total_limit": 1,
            "seed": 42,
        },
        "config_args": {
            "num_labels": 3,
            "id2label": {"0": "false", "1": "partly", "2": "true"},
            "label2id": {"false": 0, "partly": 1, "true": 2,},
        },
        "tokenizer_args": {"model_max_length": 128},
        "model_args": {},
    }

    for experiment in tqdm(experiments, desc="experiment"):
        try:
            if "base" in experiment["pretrained_model_name_or_path"]:
                args_dicts = copy.deepcopy(default_base_params)
            else:
                args_dicts = copy.deepcopy(default_large_params)

            name = [experiment["pretrained_model_name_or_path"]]

            for k, v in experiment["training_args"].items():
                args_dicts["training_args"][k] = v
                if k not in {"per_device_eval_batch_size", "logging_steps"}:
                    name.append("=".join([k, str(v)]))
            name = "___".join(name)
            _logger.info(
                "\n\n\n%s\n\n\n%s: %s\n\n\n"
                % ("-" * 64, name, json.dumps(experiment["training_args"], indent=2))
            )

            output_dir = os.path.join(base_dir, name)

            train_examples = SequenceClassificationModel.load_examples(
                train_examples_file
            )
            test_examples = SequenceClassificationModel.load_examples(
                test_examples_file
            )

            model, _, test_dataset = SequenceClassificationModel.train_from_pretrained(
                output_dir=output_dir,
                pretrained_model_name_or_path=experiment[
                    "pretrained_model_name_or_path"
                ],
                train_examples=train_examples,
                test_examples=test_examples,
                nproc=8,
                **args_dicts,
            )

            test_output = model.predict(test_dataset, predict_batch_size=1)
            test_report = get_report(test_examples, test_output)
            _logger.info("test data classification report:\n\n%s\n" % test_report)
            with open(os.path.join(output_dir, "eval_report_test_data.txt"), "w") as fo:
                fo.write(test_report)

            trial_examples = SequenceClassificationModel.load_examples(
                trial_examples_file
            )
            trial_dataset = model.create_dataset(trial_examples)
            trial_output = model.predict(trial_dataset, predict_batch_size=1)
            trial_report = get_report(trial_examples, trial_output)
            _logger.info("trial data classification report:\n\n%s\n" % trial_report)
            with open(
                os.path.join(output_dir, "eval_report_trial_data.txt"), "w"
            ) as fo:
                fo.write(trial_report)
        except:
            _logger.error("ERROR: skipping ....")
            continue
