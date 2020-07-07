import os
import copy
import json
import argparse

import wandb
import numpy as np
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split

from valerie.utils import get_logger
from valerie.datasets import Phase2Dataset
from valerie.modeling import SequenceClassificationModel, SequenceClassificationExample


os.environ["WANDB_WATCH"] = "false"  # we don't need to watch gradients
os.environ["WANDB_PROJECT"] = "valerie"  # project

models_dir = "models"
task_type = "fnc"
group_name = "initial_test_run"
base_dir = os.path.join(models_dir, task_type, group_name)

# creates dir or throws an error if it already exists
os.makedirs(base_dir)
_logger = get_logger(os.path.join(base_dir, "experiment.log"))

run_configs = [
    {"pretrained_model_name_or_path": "bert-base-cased"},
    {"pretrained_model_name_or_path": "bert-large-cased"},
    {"pretrained_model_name_or_path": "roberta-base"},
    {"pretrained_model_name_or_path": "roberta-large"},
    {"pretrained_model_name_or_path": "albert-base-v2"},
    {"pretrained_model_name_or_path": "albert-large-v2"},
]


# maybe put all the defaults in valerie utils?
def default_training_args(is_large=False):
    if is_large:
        return {
            "evaluate_during_training": True,
            "per_device_train_batch_size": 16,
            "per_device_eval_batch_size": 16,
            "gradient_accumulation_steps": 1,
            "learning_rate": 5e-5,
            "weight_decay": 0.00,
            "adam_epsilon": 1e-6,
            "max_grad_norm": 1.0,
            "num_train_epochs": 16,
            "warmup_steps": 100,
            "logging_first_step": False,
            "logging_steps": 25,
            "save_steps": 1e9,
            "save_total_limit": 1,
            "seed": 42,
        }
    else:
        return {
            "evaluate_during_training": True,
            "per_device_train_batch_size": 32,
            "per_device_eval_batch_size": 32,
            "gradient_accumulation_steps": 1,
            "learning_rate": 5e-5,
            "weight_decay": 0.00,
            "adam_epsilon": 1e-6,
            "max_grad_norm": 1.0,
            "num_train_epochs": 32,
            "warmup_steps": 50,
            "logging_first_step": False,
            "logging_steps": 15,
            "save_steps": 1e9,
            "save_total_limit": 1,
            "seed": 42,
        }


def default_config_args():
    return {
        "num_labels": 3,
        "id2label": {"0": "false", "1": "partly", "2": "true"},
        "label2id": {"false": 0, "partly": 1, "true": 2,},
    }


def default_tokenizer_args():
    return {"model_max_length": 128}


def default_model_args():
    return {}


def default_train_test_split_args():
    return {"train_size": 0.8, "random_state": 42}


def default_valerie_dataset():
    return Phase2Dataset.__name__


def construct_run_config_with_defaults(run_config):
    new_cfg = {}
    new_cfg["training_args"] = default_training_args(
        is_large="large" in run_config["pretrained_model_name_or_path"]
    )
    new_cfg["config_args"] = default_config_args()
    new_cfg["tokenizer_args"] = default_tokenizer_args()
    new_cfg["model_args"] = default_model_args()
    new_cfg["train_test_split_args"] = default_train_test_split_args()
    new_cfg["valerie_dataset"] = default_valerie_dataset()

    # update new_cfg with new values (shallow)
    for topic, topic_values in run_config.items():
        if topic in new_cfg:
            for k, v in topic_values.items():
                new_cfg[topic][k] = v
        else:
            new_cfg[topic] = topic_values

    return new_cfg


def generate_sequence_classification_examples(claims):
    examples = []
    for claim in tqdm(claims, desc="generating examples"):
        examples.append(
            SequenceClassificationExample(
                guid=claim.id, text_a=claim.claim, text_b=None, label=claim.label,
            )
        )
    return examples


def get_claims(run_config):
    name_to_dataset = {Phase2Dataset.__name__: Phase2Dataset}
    dataset_class = name_to_dataset[run_config["valerie_dataset"]]
    dataset = dataset_class.from_raw()
    claims = dataset.claims
    return claims


def get_examples(run_config):
    claims = get_claims(run_config)
    examples = generate_sequence_classification_examples(claims)

    _labels = [example.label for example in examples]
    train_examples, eval_examples, _, _ = train_test_split(
        examples, _labels, stratify=_labels, **run_config["train_test_split_args"]
    )

    return train_examples, eval_examples


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nproc", type=int, default=1)
    parser.add_argument("--test_mode", type=bool, default=False)
    parser_args = parser.parse_args()

    if parser_args.test_mode:
        os.environ["WANDB_MODE"] = "dryrun"  # don't upload experiment upload

    for i, run_config in enumerate(tqdm(run_configs, desc="run")):
        # setup run config, name, wandb integration, output dir, etc ...
        run_config = construct_run_config_with_defaults(run_config)
        run_name = group_name + "-" + str(i)
        output_dir = os.path.join(base_dir, run_name)
        os.makedirs(output_dir)
        run = wandb.init(
            name=run_name,
            tags=[task_type],
            dir=output_dir,
            group=group_name,
            reinit=True,
            allow_val_change=True,
        )

        if parser_args.test_mode:
            run_config["training_args"]["max_steps"] = 8
            run_config["training_args"]["warmup_steps"] = 2
            run_config["training_args"]["logging_steps"] = 4

        # execute the run
        with run:
            _logger.info(
                "\n\n\n%s\n%s\n%s\n%s\n\n\n",
                "-" * 80,
                run_name.center(80, "-"),
                "-" * 80,
                json.dumps(run_config, indent=2),
            )

            train_examples, eval_examples = get_examples(run_config)

            wandb.config.update(run_config)
            wandb.config.update(
                {
                    "num_train_examples": len(train_examples),
                    "num_eval_examples": len(eval_examples),
                }
            )

            SequenceClassificationModel.train_from_pretrained(
                output_dir=output_dir,
                pretrained_model_name_or_path=run_config[
                    "pretrained_model_name_or_path"
                ],
                train_examples=train_examples,
                eval_examples=eval_examples,
                training_args=run_config["training_args"],
                config_args=run_config["config_args"],
                tokenizer_args=run_config["tokenizer_args"],
                model_args=run_config["model_args"],
                exist_ok=True,
                nproc=parser_args.nproc,
            )
    _logger.info("... done :) ...")
