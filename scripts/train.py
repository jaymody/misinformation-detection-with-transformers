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

_logger = get_logger()


os.environ["WANDB_WATCH"] = "false"  # we don't need to watch gradients
os.environ["WANDB_PROJECT"] = "valerie"  # project
os.environ["WANDB_DISABLED"] = False  # set to false on test runs
# os.environ["WANDB_MODE"] = "dryrun" # uncomment this if you don't want to upload

models_dir = "models"
task_type = "fnc"
group_name = "initial_test_run"
base_dir = os.path.join(models_dir, task_type, group_name)

run_configs = [
    {"pretrained_model_name_or_path": "bert-base-cased"},
    {"pretrained_model_name_or_path": "robert-base"},
    {"pretrained_model_name_or_path": "bert-large-cased"},
    {"pretrained_model_name_or_path": "roberta-large"},
    {"pretrained_model_name_or_path": "xlnet-base-cased"},
    {"pretrained_model_name_or_path": "xlnet-large-cased"},
    {"pretrained_model_name_or_path": "t5-base"},
    {"pretrained_model_name_or_path": "t5-large"},
    {"pretrained_model_name_or_path": "albert-base-v2"},
    {"pretrained_model_name_or_path": "albert-large-v2"},
]


# maybe put all the defaults in valerie utils?
def default_training_args(is_large=False):
    if is_large:
        return {
            "evaluate_during_training": True,
            "per_device_train_batch_size": 16 / 8,
            "per_device_eval_batch_size": 16 / 8,
            "gradient_accumulation_steps": 1,
            "learning_rate": 5e-5,
            "weight_decay": 0.00,
            "adam_epsilon": 1e-6,
            "max_grad_norm": 1.0,
            "num_train_epochs": 8,
            "warmup_steps": 100 * 8,
            "logging_first_step": False,
            "logging_steps": 25 * 8,
            "save_steps": 1e9,
            "save_total_limit": 1,
            "seed": 42,
        }
    else:
        return {
            "evaluate_during_training": True,
            "per_device_train_batch_size": 32 / 16,
            "per_device_eval_batch_size": 32 / 16,
            "gradient_accumulation_steps": 1,
            "learning_rate": 5e-5,
            "weight_decay": 0.00,
            "adam_epsilon": 1e-6,
            "max_grad_norm": 1.0,
            "num_train_epochs": 16,
            "warmup_steps": 50 * 16,
            "logging_first_step": False,
            "logging_steps": 15 * 16,
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


def default_valerie_dataset_name():
    return Phase2Dataset.__name__


def construct_run_config_with_defaults(run_config):
    new_cfg = {}
    new_cfg["training_args"] = default_training_args(
        is_large="large" in new_cfg["pretrained_model_name_or_path"]
    )
    new_cfg["config_args"] = default_config_args()
    new_cfg["tokenizer_args"] = default_tokenizer_args()
    new_cfg["model_args"] = default_model_args()
    new_cfg["train_test_split_args"] = default_train_test_split_args()
    new_cfg["valerie_dataset_name"] = default_valerie_dataset_name()

    # update new_cfg with new values (shallow)
    for topic in run_config:
        if topic in new_cfg:
            for k, v in topic.items():
                new_cfg[topic][k] = v

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
        examples, _labels, stratify=_labels, **run_config["train_test_split"]
    )

    return train_examples, eval_examples


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nproc", type=int, default=1)
    nproc = parser.parse_args().nproc

    for i, run_config in enumerate(tqdm(run_configs, desc="run")):
        # setup run config, name, wandb integration, output dir, etc ...
        run_config = construct_run_config_with_defaults(run_config)
        run_name = group_name + "-" + int(i)
        output_dir = os.path.join(base_dir, run_name)
        run = wandb.init(
            name=run_name,
            tags=[task_type],
            dir=output_dir,
            group=group_name,
            reinit=True,
            allow_val_change=True,
        )

        # execute the run
        with run:
            _logger.info(
                "\n\n\n%s%s%s\n%s\n\n\n",
                "-" * 64,
                str(i) + " - " + run_name,
                "-" * 64,
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
                nproc=nproc,
            )
