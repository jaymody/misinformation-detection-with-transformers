"""Sequence classification modeling functions."""
import os
import math
import json
import random
import warnings
import logging
import multiprocessing

import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

_logger = logging.getLogger(__name__)


class InputExample:
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructor for `InputExample`.

        Parameters
        ----------
        guid : int
            Unique id for the example.
        text_a : str
            The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
        text_b : str, optional
            The untokenized text of the second sequence. Only must be specified
            for sequence pair tasks.
        label : str, optional
            The label of the example. This should be specified for train and
            dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures:
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        """Constructor for `InputFeatures`."""
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


def train(examples,
        output_dir,
        pretrained_model_name_or_path,
        config_kwargs={},
        tokenizer_kwargs={},
        model_kwargs={},

        use_cuda=True,
        cuda_device=-1,
        n_gpu=1,
        n_cpu=None,

        log_dir=None,
        logging_steps=100,
        save_steps=20000,

        batch_size=64,
        gradient_accumulation_steps=1,
        n_epochs=1,
        weight_decay=0,
        warmup_steps=0,
        warmup_ratio=0,
        learning_rate=0,
        adam_epsilon=0,
        max_grad_norm=0):
    # check that output dir exists
    if not os.path.exists(output_dir):
        raise ValueError("output_dir ({}) does not exist".format(output_dir))
    if not os.path.isdir(output_dir):
        raise ValueError("output_dir ({}) is not a directory".format(output_dir))

    # load config, model, and tokenizer
    config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **config_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, **tokenizer_kwargs)
    model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path, config=config, **model_kwargs)

    # send model to device
    device = _get_device(use_cuda, cuda_device)
    model.to(device)

    # convert examples to features
    features = _convert_examples_to_features(examples, tokenizer, config)

    # dataloader
    dataloader = DataLoader(
        features,
        sampler=RandomSampler(features), # random sampler vs shufle=True?
        batch_size=batch_size
    )

    # get total number of steps where an update occurs
    # grad accumulation lowers the number of update steps, while
    # n_epochs will increase it
    total_update_steps = len(dataloader) // gradient_accumulation_steps * n_epochs

    # setup up optimizer and scheduler
    # disable weight decay for bias params and layer normalized weights
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    warmup_steps = warmup_steps if warmup_steps else math.ceil(total_update_steps * warmup_ratio)
    optimizer = AdamW(optimizer_parameters, lr=learning_rate, eps=adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_update_steps)

    # prep model
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    model.zero_grad()
    model.train()

    # training vars
    global_step = 0
    start_epoch = 0
    start_step_in_epoch = 0
    tr_loss, logging_loss = 0, 0
    tr_acc, logging_acc = 0, 0
    tr_weightedf1, logging_weightedf1 = 0, 0
    tr_macrof1, logging_macrof1 = 0, 0

    # get starting point from initital checkpoint if specified
    if os.path.isdir(pretrained_model_name_or_path):
        # get the global step of the checkpoint
        checkpoint_suffix = pretrained_model_name_or_path.split("/")[-1].split("-")
        if len(checkpoint_suffix) > 2:
            checkpoint_suffix = checkpoint_suffix[1]
        else:
            checkpoint_suffix = checkpoint_suffix[-1]
        global_step = int(checkpoint_suffix)

        if global_step:
            start_epoch = global_step // (len(dataloader) // gradient_accumulation_steps)
            start_step_in_epoch = global_step % (len(dataloader) // gradient_accumulation_steps)

            _logger.info("   Continuing training from checkpoint, will skip to saved global_step")
            _logger.info("   Continuing training from epoch %d", start_epoch)
            _logger.info("   Continuing training from global step %d", global_step)
            _logger.info("   Will skip the first %d steps in the current epoch", start_step_in_epoch)

    # tensorboard
    tb_writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    tb_writer.add_hparams({
        "batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "n_epochs": n_epochs,
        "weight_decay": weight_decay,
        "warmup_steps": warmup_steps,
        "warmup_ratio": warmup_ratio,
        "learning_rate": learning_rate,
        "adam_epsilon": adam_epsilon,
        "max_grad_norm": max_grad_norm,
    })

    # training loop
    epoch_iterator = tqdm(range(start_epoch, n_epochs), desc="epoch")
    for _ in epoch_iterator:

        step_iterator = enumerate(tqdm(dataloader, desc="steps"))
        for step, batch in step_iterator:
            # if loading from checkpoint, skip steps already trained for in
            # the current epoch
            if start_step_in_epoch > 0:
                start_step_in_epoch -= 1
                continue

            batch = (feature.to(device) for feature in batch)
            inputs = _convert_features_to_inputs(model.base_model_prefix, batch)
            outputs = model(**inputs)

            loss = outputs[0]
            loss = loss.mean()
            loss = loss / gradient_accumulation_steps
            loss.backwards()

            preds = np.argmax(outputs[1], axis=0)

            tr_loss += loss.item()
            tr_acc += accuracy_score(inputs["labels"], preds)
            tr_weightedf1 += f1_score(inputs["labels"], preds, average="weighted")
            tr_macrof1 += f1_score(inputs["labels"], preds, average="macro")

            # don't update if gradients have not accumlated enough
            if (step + 1) % gradient_accumulation_steps != 0:
                continue

            # update weights
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            global_step += 1

            # log metrics
            if logging_steps and global_step % logging_steps == 0:
                tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                tb_writer.add_scalar("loss", (tr_loss - logging_loss) / logging_steps, global_step)
                tb_writer.add_scalar("acc", (tr_acc - logging_acc) / logging_steps, global_step)
                tb_writer.add_scalar("f1-macro", (tr_macrof1 - logging_macrof1) / logging_steps, global_step)
                tb_writer.add_scalar("f1-weighted", (tr_weightedf1 - logging_weightedf1) / logging_steps, global_step)
                logging_loss = tr_loss
                logging_acc = tr_acc
                logging_weightedf1 = tr_macrof1
                logging_macrof1 = tr_weightedf1
                # add validation loss here

            # save checkpoint
            if save_steps and global_step % save_steps == 0:
                output_dir = os.path.join(output_dir, "checkpoint-{}".format(global_step))
                _save_model(output_dir, optimizer, scheduler, model=model)

            #############################################
            ########## add early stopping here ##########
            #############################################

    return global_step, tr_loss / global_step


def predict(examples,
            pretrained_model_name_or_path,
            config_kwargs={},
            tokenizer_kwargs={},
            model_kwargs={},
            use_cuda=True,
            cuda_device=-1,
            batch_size=64):
    # load config, tokenizer, and model
    config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **config_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, **tokenizer_kwargs)
    model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path, config=config, **model_kwargs)

    # send model to device
    device = _get_device(use_cuda, cuda_device)
    model.to(device)

    # convert examples to features
    features = _convert_examples_to_features(examples, tokenizer, config)

    # data loader
    dataloader = DataLoader(
        features,
        sampler = SequentialSampler(features),
        batch_size = batch_size
    )

    # predict loop
    eval_loss = 0
    probs = []
    model.eval()
    for batch in tqdm(dataloader):
        batch = (feature.to(device) for feature in batch)

        with torch.no_grad():
            inputs = _convert_features_to_inputs(model.base_model_prefix, batch)

            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]
            eval_loss += tmp_eval_loss.mean().item()

        probs.append(logits.detach().cpu().numpy())

    eval_loss /= len(dataloader)
    preds = [np.argmax(p, axis=0) for p in probs]
    return eval_loss, probs, preds


def _save_model(output_dir, optimizer, scheduler, model):
    pass


def _convert_examples_to_features(examples, tokenizer, config):
    features = []
    for example in examples:
        encoding = tokenizer.encode_plus(example.text_a, example.text_b)
        features.append(InputFeatures(
            **encoding,
            label_id = config["label2id"][example.label] if isinstance(example.label, "str") else example.label,
        ))
    return features


def _convert_features_to_inputs(base_model_prefix, batch):
    inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}

    # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
    if base_model_prefix != "distilbert":
        inputs["token_type_ids"] = batch[2] if base_model_prefix in ["bert", "xlnet", "albert"] else None

    return inputs


def _get_device(use_cuda, cuda_device):
    """Get's device for torch."""
    if use_cuda:
        if torch.cuda.is_available():
            if cuda_device == -1:
                return torch.device("cuda") #pylint: disable=no-member
            else:
                return torch.device(f"cuda:{cuda_device}") #pylint: disable=no-member
        else:
            raise ValueError("'use_cuda' set to True when cuda is unavailable.")
    else:
        return "cpu"
