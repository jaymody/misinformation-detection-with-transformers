"""Modeling."""
import os
import json
import shutil
import logging
import collections
import dataclasses
import multiprocessing
from dataclasses import dataclass
from typing import List, Optional, Union

import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    InputFeatures,
)
from torch.utils.data.dataset import Dataset
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold

_logger = logging.getLogger(__name__)


@dataclass
class SequenceClassificationExample:
    """
    A single input example for simple sequence classification.

    Parameters
    ----------
    guid
        The claim id for the example.
    text_a
        string. The untokenized text of the first sequence. For single sequence
        tasks, only this sequence must be specified.
    text_b
        The untokenized text of the second sequence. Only must be specified for
        sequence pair tasks.
    label
        The label of the example. This should be specified for train and dev
        examples, but not for test examples.
    art_id
        The article id for the text_b string (if it exists).
    """

    guid: str
    text_a: str
    text_b: Optional[str] = None
    label: Optional[str] = None
    art_id: Optional[str] = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self), indent=2) + "\n"


class SequenceClassificationTrainingArgs(TrainingArguments):
    pass


class SequenceClassificationDataset(Dataset):
    """Sequence Classification Dataset."""

    def __init__(
        self,
        examples,
        tokenizer,
        label_list=[],
        output_mode="classification",
        nproc=1,
        cached_features_file=None,
    ):
        self.tokenizer = tokenizer
        self.output_mode = output_mode
        self.max_length = tokenizer.max_len
        self.label_map = {label: i for i, label in enumerate(label_list)}
        if cached_features_file:
            _logger.info(
                f"... loading features from cached file %s ...", cached_features_file
            )
            self.features = torch.load(cached_features_file)
        else:
            _logger.info("... converting examples to features ...")
            self.features = self.convert_examples_to_features(examples, nproc)

    def label_from_example(self, example):
        if example.label == None:
            return None
        elif self.output_mode == "classification":
            return self.label_map[example.label]
        elif self.output_mode == "regression":
            return float(example.label)
        raise KeyError(self.output_mode)

    @staticmethod
    def convert_example_to_features(_input):
        self, example = _input
        inputs = self.tokenizer.encode_plus(
            example.text_a,
            example.text_b,
            max_length=self.max_length,
            truncation=True,
            pad_to_max_length=True,
        )
        label = self.label_from_example(example)
        return InputFeatures(**inputs, label=label)

    def convert_examples_to_features(self, examples, nproc):
        all_features = []
        all_inputs = [(self, example) for example in examples]
        with multiprocessing.Pool(nproc) as pool:
            for features in tqdm(
                pool.imap(self.convert_example_to_features, all_inputs, chunksize=512),
                total=len(all_inputs),
                desc="converting examples to features",
            ):
                all_features.append(features)
        return all_features

    def save(self, cached_features_file):
        _logger.info(f".. saving features to cached file %s ...", cached_features_file)
        torch.save(self.features, cached_features_file)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        return self.features[i]


class SequenceClassificationModel:
    """Sequence classification via Transformers."""

    def __init__(self, config, tokenizer, model):
        self.config = config
        self.tokenizer = tokenizer
        self.model = model

    def create_dataset(self, examples=None, cached_features_file=None, nproc=1):
        label_list = list(self.config.label2id.values())
        return SequenceClassificationDataset(
            examples,
            tokenizer=self.tokenizer,
            label_list=label_list,
            nproc=nproc,
            cached_features_file=cached_features_file,
        )

    def train(
        self,
        train_dataset,
        test_dataset,
        training_args,
        compute_metrics=None,
        model_path=None,
    ):
        compute_metrics = compute_metrics if compute_metrics else self.compute_metrics
        self.tokenizer.save_pretrained(training_args.output_dir)
        self.config.save_pretrained(training_args.output_dir)
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics,
        )
        _global_step, _tr_loss = trainer.train(model_path=model_path)
        trainer.save_model(training_args.output_dir)
        return _global_step, _tr_loss

    def predict(self, predict_dataset, predict_batch_size=8):
        """Predict."""
        # Trainer requires an output dir to be defined, which is created
        # on init (if the folder doesn't already exist). Since we don't need
        # an output dir, we just set it to a throwaway dir in the currrent
        # directory. In addition, the default logging dir is runs, so to avoid
        # it's creation, we route the logging dir to the throwaway dir
        _temp_dir = ".valerie_tmp"
        args = SequenceClassificationTrainingArgs(
            output_dir=_temp_dir,
            logging_dir=_temp_dir,
            do_predict=True,
            per_device_eval_batch_size=predict_batch_size,  # eval batch size is used for predict
        )
        trainer = Trainer(model=self.model, args=args)
        if os.path.exists(_temp_dir):
            shutil.rmtree(_temp_dir, ignore_errors=True)
        return trainer.predict(test_dataset=predict_dataset)

    @staticmethod
    def load_examples(examples_file):
        with open(examples_file) as fi:
            examples = [
                SequenceClassificationExample(**example) for example in json.load(fi)
            ]
        return examples

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        checkpoint_dir="",
        config_args={},
        tokenizer_args={},
        model_args={},
    ):
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, **tokenizer_args
        )
        config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path, **config_args
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            os.path.join(pretrained_model_name_or_path, checkpoint_dir)
            if checkpoint_dir
            else pretrained_model_name_or_path,
            config=config,
            **model_args,
        )
        return cls(config, tokenizer, model)

    @classmethod
    def train_from_pretrained(
        cls,
        output_dir,
        pretrained_model_name_or_path,
        train_examples,
        test_examples=None,
        data_args={},
        training_args={},
        config_args={},
        tokenizer_args={},
        model_args={},
        compute_metrics=None,
        nproc=1,
    ):
        os.makedirs(output_dir)
        with open(os.path.join(output_dir, "data_args.json"), "w") as fo:
            json.dump(data_args, fo, indent=2)
        with open(os.path.join(output_dir, "training_args.json"), "w") as fo:
            json.dump(training_args, fo, indent=2)
        with open(os.path.join(output_dir, "config_args.json"), "w") as fo:
            json.dump(config_args, fo, indent=2)
        with open(os.path.join(output_dir, "tokenizer_args.json"), "w") as fo:
            json.dump(tokenizer_args, fo, indent=2)
        with open(os.path.join(output_dir, "model_args.json"), "w") as fo:
            json.dump(model_args, fo, indent=2)

        model = cls.from_pretrained(
            pretrained_model_name_or_path,
            config_args=config_args,
            tokenizer_args=tokenizer_args,
            model_args=model_args,
        )

        train_dataset = model.create_dataset(examples=train_examples, nproc=nproc)
        test_dataset = model.create_dataset(examples=test_examples, nproc=nproc)

        training_args = SequenceClassificationTrainingArgs(
            output_dir=output_dir, logging_dir=output_dir, **training_args
        )

        _global_step, _tr_loss = model.train(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            training_args=training_args,
            model_path=pretrained_model_name_or_path,
            compute_metrics=compute_metrics,
        )

        return model, train_dataset, test_dataset

    @classmethod
    def train_kfold_from_pretrained(
        cls,
        output_dir,
        pretrained_model_name_or_path,
        examples,
        data_args={},
        training_args={},
        config_args={},
        tokenizer_args={},
        model_args={},
        compute_metrics=None,
        nproc=1,
    ):
        if "StratifiedKFold" not in data_args:
            raise ValueError(
                "key StratifiedKFold must be in data_args and contain all"
                "kwargs for sklearn.model_selection.StratifiedKFold."
            )

        # create output dir and save arg dicts
        os.makedirs(output_dir)
        args_dict = {
            "data_args.json": data_args,
            "training_args.json": training_args,
            "config_args.json": config_args,
            "tokenizer_args.json": tokenizer_args,
            "model_args.json": model_args,
        }
        for k, v in args_dict.items():
            with open(os.path.join(output_dir, k), "w") as fo:
                json.dump(v, fo, indent=2)

        # kfold
        labels = [example.label for example in examples]
        skf = StratifiedKFold(
            data_args["StratifiedKFold"]["n_splits"],
            shuffle=data_args["StratifiedKFold"]["shuffle"],
            random_state=data_args["StratifiedKFold"]["random_state"],
        )
        predictions = {}
        for k, (train_index, test_index) in tqdm(
            enumerate(skf.split(examples, labels)),
            total=data_args["StratifiedKFold"]["n_splits"],
            desc="fold",
        ):
            # create fold dir and save arg dicts
            fold_dir = os.path.join(output_dir, "fold-{}".format(k))
            os.makedirs(fold_dir)
            for k, v in args_dict.items():
                with open(os.path.join(fold_dir, k), "w") as fo:
                    json.dump(v, fo, indent=2)

            # init model
            model = cls.from_pretrained(
                pretrained_model_name_or_path,
                config_args=config_args,
                tokenizer_args=tokenizer_args,
                model_args=model_args,
            )

            # convert examples to features datasets
            train_examples = [examples[i] for i in train_index]
            test_examples = [examples[i] for i in test_index]
            train_dataset = model.create_dataset(examples=train_examples, nproc=nproc)
            test_dataset = model.create_dataset(examples=test_examples, nproc=nproc)

            # train
            _global_step, _tr_loss = model.train(
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                training_args=SequenceClassificationTrainingArgs(
                    output_dir=fold_dir, logging_dir=fold_dir, **training_args
                ),
                model_path=pretrained_model_name_or_path,
                compute_metrics=compute_metrics,
            )

            # predict
            predict_output = model.predict(
                predict_dataset=test_dataset,
                predict_batch_size=training_args["per_device_eval_batch_size"],
            )

            for example, prob in zip(test_examples, predict_output.predictions):
                assert example.guid not in predictions
                predictions[example.guid] = prob

        return predictions

    @staticmethod
    def compute_metrics(results):
        labels = results.label_ids
        probs = results.predictions
        preds = [np.argmax(v) for v in probs]
        report = classification_report(labels, preds, output_dict=True)

        metrics = {}
        metrics["accuracy"] = report.pop("accuracy")

        macro_avg = report.pop("macro avg")
        metrics["f1_macro"], metrics["recall_macro"], metrics["precision_macro"] = (
            macro_avg["f1-score"],
            macro_avg["recall"],
            macro_avg["precision"],
        )

        weighted_avg = report.pop("weighted avg")
        (
            metrics["f1_weighted"],
            metrics["recall_weighted"],
            metrics["precision_weighted"],
        ) = (
            weighted_avg["f1-score"],
            weighted_avg["recall"],
            weighted_avg["precision"],
        )

        for k, v in report.items():
            metrics["f1_{}".format(k)] = v["f1-score"]
            metrics["recall_{}".format(k)] = v["recall"]
            metrics["precision_{}".format(k)] = v["precision"]

        return metrics


class ClaimantModel:
    """Claimant modeling based on claimaint truth history."""

    id2label = {0: "false", 1: "partly", 2: "true"}

    def __init__(self, model={}):
        self.model = model

    @classmethod
    def from_pretrained(cls, model_file):
        with open(model_file) as fi:
            return cls(**json.load(fi))

    def save_pretrained(self, model_file):
        with open(model_file, "w") as fo:
            json.dump(self.__dict__, fo)

    def train(self, claims, min_threshold=10):
        self.model = self.analyze(claims, min_threshold=min_threshold)

    def predict(self, claim):
        try:
            entry = self.model[claim.claimant]

            probs = np.zeros(len(self.id2label))
            for i, label in self.id2label.items():
                probs[i] = entry[label] / entry["total"]

            return probs
        except KeyError:
            return None

    def num_examples(self, claim):
        try:
            return self.model[claim.claimant]["total"]
        except KeyError:
            return None

    def score(self, claim):
        try:
            return self.model[claim.claimant]["score"]
        except KeyError:
            return None

    @classmethod
    def analyze(cls, claims, min_threshold=10, return_df=False):
        model = collections.defaultdict(
            lambda: {"false": 0, "partly": 0, "true": 0, "score": 0, "total": 0,}
        )

        for claim in claims:
            model[claim.claimant][cls.id2label[claim.label]] += 1
            model[claim.claimant]["total"] += 1
            model[claim.claimant]["score"] += claim.label / 2

        for v in model.values():
            v["score"] /= v["total"]

        model = {
            k: v
            for k, v in sorted(model.items(), key=lambda item: item[1]["score"])
            if v["total"] > min_threshold
        }

        if return_df:
            return pd.DataFrame.from_dict(model, orient="index")
        else:
            return model


class SourceModel:
    """Source modeling based on article sources."""

    def __init__(self, model={}, max_threshold=100, min_threshold=3):
        self.model = model
        self.max_threshold = max_threshold
        self.min_threshold = min_threshold

    @classmethod
    def from_pretrained(cls, model_file):
        with open(model_file) as fi:
            return cls(**json.load(fi))

    def save_pretrained(self, model_file):
        with open(model_file, "w") as fo:
            json.dump(self.__dict__, fo)

    def train(self, articles, min_threshold=0):
        self.model = self.analyze(articles, min_threshold=min_threshold)

    def predict(self, article):
        try:
            value = self.model[article.source]
            if value > self.max_threshold:
                return 1.0
            elif value < self.min_threshold:
                return 0.0
            else:
                return value / self.max_threshold
        except KeyError:
            return None

    def num_examples(self, article):
        try:
            return self.model[article.source]
        except KeyError:
            return None

    @classmethod
    def analyze(cls, articles, min_threshold=0):
        model = collections.defaultdict(int)
        for article in articles:
            model[article.source] += 1

        model = {k: v for k, v in model.items() if v > min_threshold}
        return model
