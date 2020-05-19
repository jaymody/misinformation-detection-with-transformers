import json
import collections

import numpy as np
import pandas as pd


class SequenceClassificationModel:
    pass


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
