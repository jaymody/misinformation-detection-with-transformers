import os
import json
import random
import argparse
import collections
from dataclasses import dataclass

import torch
import numpy as np
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    InputExample,
)
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.tensorboard import SummaryWriter

from valerie.utils import get_logger
from valerie.datasets import BasicDataset

_logger = get_logger()


@dataclass
class DataArguments():
    """Data arguments."""

    examples_file: str
    cached_train_features_file: str = None
    cached_test_features_file: str = None
    train_size: float = 0.95
    random_state: int = None


def get_args_files(output_dir):
    if not os.path.exists(output_dir):
        raise ValueError(f"output dir does not exists: {output_dir}")

    args_files = {
        "data_args_file": os.path.join(output_dir, "data_args.json"),
        "config_args_file": os.path.join(output_dir, "config_args.json"),
        "tokenizer_args_file": os.path.join(output_dir, "tokenizer_args.json"),
        "model_args_file": os.path.join(output_dir, "model_args.json"),
        "training_args_file": os.path.join(output_dir, "training_args.json"),
    }
    for k, v in args_files.items():
        if not os.path.exists(v):
            raise ValueError(f"could not find {k}: {v}")

    return args_files


def from_pretrained(pretrained_model_name_or_path, config_args={}, tokenizer_args={}, model_args={}):
    config = AutoConfig.from_pretrained(
        pretrained_model_name_or_path,
        **config_args
    )
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        **tokenizer_args
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path,
        config=config,
        **model_args
    )
    return config, tokenizer, model


def load_examples(examples_file):
    with open(examples_file) as fi:
        examples = [InputExample(**example) for example in json.load(fi)]
    return examples


def train_test_split(examples, train_size=0.95, random_state=None):
    _examples = collections.defaultdict(list)
    for example in tqdm(examples):
        _examples[example.guid].append(example)
    _examples = list(_examples.values())

    num_total_claims = len(_examples)
    num_total_examples = len(examples)
    assert len(examples) == sum([len(example) for example in _examples])

    sss = StratifiedShuffleSplit(
        n_splits=1,
        train_size=train_size,
        random_state=random_state
    )
    labels = [example[0].label for example in _examples]
    train_index, test_index = list(sss.split(_examples, labels))[0]

    training_examples = [_examples[idx] for idx in train_index]
    num_training_claims = len(training_examples)
    training_examples = [example for example_list in training_examples for example in example_list]
    num_training_examples = len(training_examples)

    testing_examples = [_examples[idx] for idx in test_index]
    num_testing_claims = len(testing_examples)
    testing_examples = [example for example_list in testing_examples for example in example_list]
    num_testing_examples = len(testing_examples)

    _logger.info("Num Total Claims:\t\t%d", num_total_claims)
    _logger.info("Num Training Claims:\t%d", num_training_claims)
    _logger.info("Num Testing Claims:\t%d", num_testing_claims)
    _logger.info("")
    _logger.info("Num Total Examples:\t%d", num_total_examples)
    _logger.info("Num Training Examples:\t%d", num_training_examples)
    _logger.info("Num Testing Examples:\t%d", num_testing_examples)
    _logger.info("")

    train_labels_count = collections.Counter([labels[idx] for idx in train_index])
    test_labels_count = collections.Counter([labels[idx] for idx in test_index])
    _logger.info("Train Labels Count:\t%s", str(dict(train_labels_count)))
    _logger.info("Test Labels Count:\t\t%s", str(dict(test_labels_count)))

    return training_examples, testing_examples


def compute_metrics(results):
    labels = results.label_ids
    probs = results.predictions
    preds = [np.argmax(v) for v in probs]
    report = classification_report(labels, preds, output_dict=True)

    metrics = {}
    metrics["accuracy"] = report.pop("accuracy")
    metrics["f1_macro"] = report.pop("macro avg")["f1-score"]
    metrics["f1_weighted"] = report.pop("weighted avg")["f1-score"]
    for k, v in report.items():
        metrics["f1_label_{}".format(k)] = v["f1-score"]

    return metrics


def train(output_dir,
        pretrained_model_name_or_path,
        data_args_file,
        training_args_file,
        config_args_file="",
        tokenizer_args_file="",
        model_args_file="",
        label_list=[],
        compute_metrics_fn=compute_metrics,
        nproc=1):
    """Train sequence classifier."""
    config_args = {}
    tokenizer_args = {}
    model_args = {}
    if config_args_file:
        with open(config_args_file) as fi:
            config_args = json.load(fi)
    if tokenizer_args_file:
        with open(tokenizer_args_file) as fi:
            tokenizer_args = json.load(fi)
    if model_args_file:
        with open(model_args_file) as fi:
            model_args = json.load(fi)

    with open(data_args_file) as fi:
        data_args = DataArguments(**json.load(fi))
    with open(training_args_file) as fi:
        training_args = TrainingArguments(output_dir=output_dir, **json.load(fi))

    config, tokenizer, model = from_pretrained(
        pretrained_model_name_or_path,
        config_args,
        tokenizer_args,
        model_args
    )

    label_list = label_list if label_list else list(config.label2id.values())

    train_dataset = None
    test_dataset = None
    # load from cached files, else load from examples file
    if data_args.cached_train_features_file:
        train_dataset = BasicDataset(
            None,
            tokenizer=tokenizer,
            label_list=label_list,
            nproc=nproc,
            cached_features_file=data_args.cached_train_features_file
        )
        if data_args.cached_test_features_file:
            test_dataset = BasicDataset(
                None,
                tokenizer=tokenizer,
                label_list=label_list,
                nproc=nproc,
                cached_features_file=data_args.cached_test_features_file
            )
    else:
        training_examples = load_examples(data_args.examples_file)
        testing_examples = None
        if data_args.train_size:
            training_examples, testing_examples = train_test_split(
                training_examples,
                train_size=data_args.train_size,
                random_state=data_args.random_state
            )
            test_dataset = BasicDataset(
                testing_examples,
                tokenizer=tokenizer,
                label_list=label_list,
                nproc=nproc,
            )

        train_dataset = BasicDataset(
            training_examples,
            tokenizer=tokenizer,
            label_list=label_list,
            nproc=nproc,
        )

    hparams_dict = {
        "per_gpu_train_batch_size": training_args.per_gpu_train_batch_size,
        "per_gpu_eval_batch_size": training_args.per_gpu_eval_batch_size,
        "train_batch_size": training_args.train_batch_size,
        "eval_batch_size": training_args.eval_batch_size,
        "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
        "learning_rate": training_args.learning_rate,
        "weight_decay": training_args.weight_decay,
        "adam_epsilon": training_args.adam_epsilon,
        "max_grad_norm": training_args.max_grad_norm,
        "num_train_epochs": training_args.num_train_epochs,
        "max_steps": training_args.max_steps,
        "warmup_steps": training_args.warmup_steps,
        "seed": training_args.seed,
        "fp16": training_args.fp16,
        "fp16_opt_level": training_args.fp16_opt_level,
    }

    tb_writer = SummaryWriter(log_dir=training_args.output_dir)
    tb_writer.add_hparams(hparams_dict, {})
    _example_input_to_model = torch.zeros(
        [training_args.train_batch_size, tokenizer.max_len],
        dtype=torch.long
    )
    tb_writer.add_graph(model, _example_input_to_model)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        tb_writer=tb_writer,
    )

    _global_step, _tr_loss = trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train sequence classifier.")
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--pretrained_model_name_or_path", type=str)
    parser.add_argument("--nproc", type=int, default=1)

    args = parser.parse_args()
    args_files = get_args_files(args.output_dir)
    train(**args_files, **args.__dict__)
