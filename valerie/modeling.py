import json
import logging
import collections
import multiprocessing

import numpy as np
import pandas as pd

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
    InputFeatures,
)
from torch.utils.data.dataset import Dataset
from sklearn.metrics import classification_report

_logger = logging.getLogger(__name__)


class SequenceClassificationInputExample(InputExample):
    pass


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

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        config_args={},
        tokenizer_args={},
        model_args={},
    ):
        config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path, **config_args
        )
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, **tokenizer_args
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path, config=config, **model_args
        )
        return cls(config, tokenizer, model)

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
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics,
        )
        return trainer.train(model_path=model_path)

    def predict(self, predict_dataset, predict_batch_size=8):
        """Predict."""
        args = SequenceClassificationTrainingArgs(
            output_dir="", do_predict=True, eval_batch_size=predict_batch_size
        )
        trainer = Trainer(model=self.model, args=args)
        return trainer.predict(test_dataset=predict_dataset)

    @staticmethod
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
