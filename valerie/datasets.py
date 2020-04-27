import time
import logging
import multiprocessing
from dataclasses import dataclass, field

import torch
from tqdm import tqdm
from transformers import InputFeatures
from torch.utils.data.dataset import Dataset

_logger = logging.getLogger(__name__)


class BasicDataset(Dataset):
    """Basic Dataset."""

    def __init__(self, examples, tokenizer, label_list=[], output_mode="classification", nproc=1, cached_features_file=None):
        self.nproc = nproc
        self.tokenizer = tokenizer
        self.output_mode = output_mode
        self.max_length = tokenizer.max_len
        self.label_map = {label: i for i, label in enumerate(label_list)}
        if cached_features_file:
            _logger.info(f"... loading features from cached file %s", cached_features_file)
            self.features = torch.load(cached_features_file)
        else:
            _logger.info("... converting examples to features ...")
            self.features = self.convert_examples_to_features(examples)

    def label_from_example(self, example):
        if example.label == None:
            return None
        elif self.output_mode == "classification":
            return self.label_map[example.label]
        elif self.output_mode == "regression":
            return float(example.label)
        raise KeyError(self.output_mode)

    def convert_examples_to_features(self, examples):
        all_features = []
        pool = multiprocessing.Pool(self.nproc)
        for features in tqdm(pool.map(self._convert_example_to_features, examples), total=len(examples)):
            all_features.append(features)
        return all_features

    def _convert_example_to_features(self, example):
        inputs = self.tokenizer.encode_plus(
            example.text_a,
            example.text_b,
            max_length=self.max_length,
            pad_to_max_length=True
        )
        label = self.label_from_example(example)
        return InputFeatures(**inputs, label=label)

    def save(self, cached_features_file):
        _logger.info(f".. saving features to cached file %s", cached_features_file)
        torch.save(self.features, cached_features_file)

    def  __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        return self.features[i]
