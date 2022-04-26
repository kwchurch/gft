#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning a 🤗 Transformers model on token classification tasks (NER, POS, CHUNKS) relying on the accelerate library
without using a Trainer.
"""

# based on https://github.com/huggingface/transformers/blob/master/examples/pytorch/token-classification/run_ner_no_trainer.py

import gft
import sys,os,shutil
import seqeval,time,sys

import argparse
import logging
import math
import os
import random

import datasets
import torch
from datasets import ClassLabel, load_dataset, load_metric
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm

from gft.gft_internals import parse_eqn
from gft.gft_internals.gft_util import intern_labels, parse_model_specification, parse_metric_specification, better, checkpoint_filename, get_arg, set_arg

import transformers
from accelerate import Accelerator

from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AdamW,
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    set_seed,
)
from transformers.utils.versions import require_version


logger = logging.getLogger(__name__)
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/token-classification/requirements.txt")

# You should update this to your particular problem to have better documentation of `model_type`
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


# def parse_args():
#     parser = argparse.ArgumentParser(
#         description="Finetune a transformers model on a text classification task (NER) with accelerate library"
#     )
#     parser.add_argument(
#         "--dataset_name",
#         type=str,
#         default=None,
#         help="The name of the dataset to use (via the datasets library).",
#     )
#     parser.add_argument(
#         "--dataset_config_name",
#         type=str,
#         default=None,
#         help="The configuration name of the dataset to use (via the datasets library).",
#     )
#     parser.add_argument(
#         "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
#     )
#     parser.add_argument(
#         "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
#     )
#     parser.add_argument(
#         "--text_column_name",
#         type=str,
#         default=None,
#         help="The column name of text to input in the file (a csv or JSON file).",
#     )
#     parser.add_argument(
#         "--label_column_name",
#         type=str,
#         default=None,
#         help="The column name of label to input in the file (a csv or JSON file).",
#     )
#     parser.add_argument(
#         "--max_length",
#         type=int,
#         default=128,
#         help=(
#             "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
#             " sequences shorter will be padded if `--pad_to_max_length` is passed."
#         ),
#     )
#     parser.add_argument(
#         "--pad_to_max_length",
#         action="store_true",
#         help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
#     )
#     parser.add_argument(
#         "--model_name_or_path",
#         type=str,
#         help="Path to pretrained model or model identifier from huggingface.co/models.",
#         required=True,
#     )
#     parser.add_argument(
#         "--config_name",
#         type=str,
#         default=None,
#         help="Pretrained config name or path if not the same as model_name",
#     )
#     parser.add_argument(
#         "--tokenizer_name",
#         type=str,
#         default=None,
#         help="Pretrained tokenizer name or path if not the same as model_name",
#     )
#     parser.add_argument(
#         "--per_device_train_batch_size",
#         type=int,
#         default=8,
#         help="Batch size (per device) for the training dataloader.",
#     )
#     parser.add_argument(
#         "--per_device_eval_batch_size",
#         type=int,
#         default=8,
#         help="Batch size (per device) for the evaluation dataloader.",
#     )
#     parser.add_argument(
#         "--learning_rate",
#         type=float,
#         default=5e-5,
#         help="Initial learning rate (after the potential warmup period) to use.",
#     )
#     parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
#     parser.add_argument("--epochs", type=int, default=3, help="Total number of training epochs to perform.")
#     parser.add_argument(
#         "--max_train_steps",
#         type=int,
#         default=None,
#         help="Total number of training steps to perform. If provided, overrides epochs.",
#     )
#     parser.add_argument(
#         "--gradient_accumulation_steps",
#         type=int,
#         default=1,
#         help="Number of updates steps to accumulate before performing a backward/update pass.",
#     )
#     parser.add_argument(
#         "--lr_scheduler_type",
#         type=SchedulerType,
#         default="linear",
#         help="The scheduler type to use.",
#         choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
#     )
#     parser.add_argument(
#         "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
#     )
#     parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
#     parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
#     parser.add_argument(
#         "--model_type",
#         type=str,
#         default=None,
#         help="Model type to use if training from scratch.",
#         choices=MODEL_TYPES,
#     )
#     parser.add_argument(
#         "--label_all_tokens",
#         action="store_true",
#         help="Setting labels of all special tokens to -100 and thus PyTorch will ignore them.",
#     )
#     parser.add_argument(
#         "--return_entity_level_metrics",
#         action="store_true",
#         help="Indication whether entity level metrics are to be returner.",
#     )
#     parser.add_argument(
#         "--task_name",
#         type=str,
#         default="ner",
#         choices=["ner", "pos", "chunk"],
#         help="The name of the task.",
#     )
#     parser.add_argument(
#         "--debug",
#         action="store_true",
#         help="Activate debug mode and run training only with a subset of data.",
#     )
#     args = parser.parse_args()

#     # Sanity checks
#     if args.task_name is None and args.train_file is None and args.validation_file is None:
#         raise ValueError("Need either a task name or a training/validation file.")
#     else:
#         if args.train_file is not None:
#             extension = args.train_file.split(".")[-1]
#             assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
#         if args.validation_file is not None:
#             extension = args.validation_file.split(".")[-1]
#             assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

#     if args.output_dir is not None:
#         os.makedirs(args.output_dir, exist_ok=True)

#     return args

def fit(args, eqn, accelerator, raw_datasets):
    print('calling fit in fit_for_classify_tokens.py')
    assert eqn['eqn_type'] == parse_eqn.eqn_types['classify_tokens'], \
        'fit_for_classify_tokens is for eqn_type: classify_tokens, but eqn_type is: ' + str(eqn['eqn_type'])


    # If passed along, set the training seed now.
    seed = get_arg(args, 'seed', default=None)
    print('seed: ' + str(seed))
    if not seed is None:
        set_seed(seed)

    # Sample hyper-parameters for learning rate, batch size, seed and a few other HPs
    lr = get_arg(args, 'learning_rate', default=5e-5)
    num_epochs = int(get_arg(args, 'num_train_epochs', default=15))
    # correct_bias = config["correct_bias"]
    batch_size = int(get_arg(args, 'per_device_train_batch_size', default=16))

    # # If passed along, set the training seed now.
    # print('args.seed: ' + str(args.seed))
    # if args.seed is not None:
    #     set_seed(args.seed)

    # # Sample hyper-parameters for learning rate, batch size, seed and a few other HPs
    # lr = config["lr"]
    # num_epochs = config["num_epochs"]
    # correct_bias = config["correct_bias"]
    # seed = config["seed"]
    # batch_size = config["batch_size"]

    x_field_names = eqn['x_field_names']
    y_field_names = eqn['y_field_names']
    interned_labels,label_list = intern_labels(raw_datasets, y_field_names, args)
    num_labels = len(label_list)

    label_to_id = {i:i for i in range(num_labels)}

    assert len(y_field_names) == 1, 'classify_tokens is currently limited to just one y variable: ' + str(y_field_names)
    assert len(x_field_names) in [1,2], 'classify_tokens is currently limited to just one or two x variables: ' + str(x_field_names)

    print('label_list: ' + str(label_list))
    print('interned_labels: ' + str(interned_labels))
    print('num_labels: ' + str(num_labels))
    print('label_to_id: ' + str(label_to_id))

    def id_ify(lab):
        if lab in interned_labels:
            return interned_labels[lab]
        else:
            return label_to_id[lab]

    provider, model_key = parse_model_specification(args)
    assert not provider == 'PaddlePaddle', 'PaddlePaddle models are not supported (yet)'  # stub to flesh out later

    metric_provider, metric_key = parse_metric_specification(args)
    assert not metric_provider == 'PaddlePaddle', 'PaddlePaddle metrics are not supported (yet)'  # stub to flesh out later

    if metric_key is None:
        print('load_metric: seqeval')
        val_metric = load_metric("seqeval")
        train_metric = load_metric("seqeval")
    else:
        print('load_metric: ' + metric_key)
        val_metric = load_metric(*metric_key.split(','))
        train_metric = load_metric(*metric_key.split(','))

    device = accelerator.device
    # end of borrowing from old version

    # from run_ner_no_trainer.py
    # def main():
    #     args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO)
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    # if args.config_name:
    #     config = AutoConfig.from_pretrained(args.config_name, num_labels=num_labels)
    # if model_key:

    config = AutoConfig.from_pretrained(model_key, num_labels=num_labels)

    # else:
    #     config = CONFIG_MAPPING[args.model_type]()
    #     logger.warning("You are instantiating a new config instance from scratch.")

    # tokenizer_name_or_path = args.tokenizer_name if args.tokenizer_name else model_key
    tokenizer_name_or_path = model_key
    if not tokenizer_name_or_path:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if config.model_type in {"gpt2", "roberta"}:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=True, add_prefix_space=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=True)

    if model_key:
        model = AutoModelForTokenClassification.from_pretrained(model_key, from_tf=bool(".ckpt" in model_key), config=config)
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForTokenClassification.from_config(config)

    model.resize_token_embeddings(len(tokenizer))

    # Preprocessing the datasets.
    # First we tokenize all the texts.

    padding = "max_length" if get_arg(args, 'pad_to_max_length', default=False) else False

    # Tokenize all texts and align the labels with them.

    def maybe_split(s):
        if get_arg(args, 'is_split_into_words', default=False): return s
        else: return [e.split() for e in s]

    def tokenize_and_align_labels(examples):

        # print('args.is_split_into_words: ' + str(args.is_split_into_words))

        max_length=get_arg(args, 'max_length', default=128)

        if len(x_field_names) == 1:
            tokenized_inputs = tokenizer(
                maybe_split(examples[x_field_names[0]]),
                max_length=max_length,
                padding=padding,
                truncation=True,
                # We use this argument because the texts in our dataset are lists of words (with a label for each word).
                is_split_into_words=True)
        elif len(x_field_names) == 2:
            tokenized_inputs = tokenizer(
                maybe_split(examples[x_field_names[0]]),
                maybe_split(examples[x_field_names[1]]),
                max_length=max_length,
                padding=padding,
                truncation=True,
                # We use this argument because the texts in our dataset are lists of words (with a label for each word).
                is_split_into_words=True)
        else: assert False, 'should not get here'

        label_all_tokens = get_arg(args, 'label_all_tokens', default=False)
        labels = []
        label_column_name = y_field_names[0]
        for i, lab in enumerate(examples[label_column_name]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []

            if get_arg(args, 'is_split_into_words', default=False): label = lab
            else: label = lab.split()

            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    #  print('word_idx: ' + str(word_idx) + ' of ' + str(len(label)))
                    label_ids.append(id_ify(label[word_idx]))
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    # print('word_idx: ' + str(word_idx) + ' of ' + str(len(label)))
                    label_ids.append(id_ify(label[word_idx]) if label_all_tokens else -100)
                previous_word_idx = word_idx

            labels.append(label_ids)
        tokenized_inputs["labels"] = labels

        return tokenized_inputs

    def colnames(split):
        try:
            return split.column_names
        except:
            return None

    processed_raw_datasets = raw_datasets.map(
        tokenize_and_align_labels,
        batched=True,
        # remove_columns=raw_datasets["train"].column_names,
        remove_columns=colnames(raw_datasets["train"]),
        desc="Running tokenizer on dataset",
    )

    assert 'val' in processed_raw_datasets, 'cannot find validation set'
    train_dataset = processed_raw_datasets["train"]
    eval_dataset = processed_raw_datasets["val"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    pad_to_max_length = get_arg(args, 'pad_to_max_length', default=False)
    if pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorForTokenClassification` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorForTokenClassification(
            tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None)
            )

    bs = get_arg(args, 'per_device_train_batch_size', default=32)
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=bs)

    bs = get_arg(args, 'per_device_eval_batch_size', default=32)
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=bs)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": get_arg(args, 'weight_decay', default=0.0),
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)

    # Use the device given by the `accelerator` object.
    device = accelerator.device
    model.to(device)

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader)

    # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
    # shorter in multiprocess)

    # Scheduler and math around the number of training steps.
    gas = get_arg(args, 'gradient_accumulation_steps', default=1)
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gas)
    epochs = int(get_arg(args, 'num_train_epochs', default=15))
    max_train_steps = get_arg(args, 'max_train_steps', default=None)
    if max_train_steps is None:
        max_train_steps = epochs * num_update_steps_per_epoch
    else:
        epochs = int(math.ceil(max_train_steps / num_update_steps_per_epoch))

    # num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    # if args.max_train_steps is None:
    #     args.max_train_steps = args.epochs * num_update_steps_per_epoch
    # else:
    #     args.epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name=get_arg(args, 'lr_scheduler_type', default="linear"),
        optimizer=optimizer,
        num_warmup_steps=get_arg(args, 'num_warmup_steps', default=0),
        num_training_steps=max_train_steps,
    )


    # lr_scheduler = get_scheduler(
    #     name=args.lr_scheduler_type,
    #     optimizer=optimizer,
    #     num_warmup_steps=args.num_warmup_steps,
    #     num_training_steps=args.max_train_steps,
    # )

    # Metrics
    # https://raw.githubusercontent.com/huggingface/datasets/1.12.1/metrics/seqeval/seqeval.py
    # https://github.com/chakki-works/seqeval
    metric = load_metric("seqeval")

    def get_labels(predictions, references):
        # Transform predictions and references tensos to numpy arrays
        if device.type == "cpu":
            y_pred = predictions.detach().clone().numpy()
            y_true = references.detach().clone().numpy()
        else:
            y_pred = predictions.detach().cpu().clone().numpy()
            y_true = references.detach().cpu().clone().numpy()

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(pred, gold_label) if l != -100]
            for pred, gold_label in zip(y_pred, y_true)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(pred, gold_label) if l != -100]
            for pred, gold_label in zip(y_pred, y_true)
        ]
        return true_predictions, true_labels

    def compute_metrics():
        results = metric.compute()
        if get_arg(args, 'return_entity_level_metrics', default=False):
            # Unpack nested dictionaries
            final_results = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    for n, v in value.items():
                        final_results[f"{key}_{n}"] = v
                else:
                    final_results[key] = value
            return final_results
        else:
            return {
                "precision": results["overall_precision"],
                "recall": results["overall_recall"],
                "f1": results["overall_f1"],
                "accuracy": results["overall_accuracy"],
            }

    # Train!
    total_batch_size = bs * accelerator.num_processes * gas
    # total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps


    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {epochs}")
    logger.info(f"  Instantaneous batch size per device = {bs}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {gas}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    completed_steps = 0

    # logger.info("***** Running training *****")
    # logger.info(f"  Num examples = {len(train_dataset)}")
    # logger.info(f"  Num Epochs = {args.epochs}")
    # logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    # logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    # logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    # logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    # progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process) 
    completed_steps = 0

    best_so_far = None
    time0 = time.time()
    for epoch in range(epochs):
        print('about to train epoch %d: %0.0f sec' % (epoch, time.time() - time0))
        model.train()
        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            loss = loss / gas
            accelerator.backward(loss)
            if step % gas == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                # progress_bar.update(1)
                completed_steps += 1

            if completed_steps >= max_train_steps:
                break

        print('about to eval epoch %d: %0.0f sec' % (epoch, time.time() - time0))
        model.eval()
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            labels = batch["labels"]
            if not pad_to_max_length:  # necessary to pad predictions and labels for being gathered
                predictions = accelerator.pad_across_processes(predictions, dim=1, pad_index=-100)
                labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)

            predictions_gathered = accelerator.gather(predictions)
            labels_gathered = accelerator.gather(labels)
            preds, refs = get_labels(predictions_gathered, labels_gathered)

            metric.add_batch(
                predictions=preds,
                references=refs,
            )  # predictions and preferences are expected to be a nested list of labels, not label_ids

        eval_metric = metric.compute()
        logger.info(f"epoch {epoch}: {eval_metric} {time.time() - time0} seconds")
        accelerator.print('%0.0f seconds: epoch %d validation %s' % (time.time() - time0, epoch, str(eval_metric)))

        fn = checkpoint_filename(args, model_key, epoch, False)
        b,best = better(args, eval_metric, best_so_far)
        if b:
            best_so_far = best
            print('best_so_far: %f' % (best_so_far))
            fn = checkpoint_filename(args, model_key, epoch, True)
        if not get_arg(args, 'output_dir') is None:
            prev = checkpoint_filename(args, model_key, epoch-1, False)
            try:
                if os.path.exists(prev): shutil.rmtree(prev)
            except:
                print('failed to delete: ' + str(prev), sys.stderr)
            model.save_pretrained(fn)

    # if args.checkpoint is not None:
    #     accelerator.wait_for_everyone()
    #     unwrapped_model = accelerator.unwrap_model(model)
    #     unwrapped_model.save_pretrained(args.checkpoint, save_function=accelerator.save)
