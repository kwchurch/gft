import gft
# based on https://github.com/huggingface/transformers/blob/master/examples/pytorch/text-classification/run_glue_no_trainer.py

# convention: _hf is from HuggingFace, and _pd is from PaddleHub

# import pdb

import sys,os,shutil,torch,time
time0 = time.time()

from torch.utils.data import DataLoader
import numpy as np
from gft.gft_internals import parse_eqn
from gft.gft_internals.gft_util import intern_labels, parse_model_specification, parse_metric_specification, parse_dataset_specification, better, checkpoint_filename, get_arg, set_arg

import logging
import math
import os
import random

import datasets,transformers
from datasets import load_dataset, load_metric

# from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm


#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
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
Fine-tuning the library models for question answering using a slightly adapted version of the 🤗 Trainer.
"""
# You can also adapt this script on your own question answering task. Pointers for this are left as comments.

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
from datasets import load_dataset, load_metric

import transformers
from gft.gft_internals.trainer_qa import QuestionAnsweringTrainer
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from gft.gft_internals.utils_qa import postprocess_qa_predictions


# kwc: This requires a newer version than what is available from pip

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.18.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/question-answering/requirements.txt")

logger = logging.getLogger(__name__)

# @dataclass
# class ModelArguments:
#     """
#     Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
#     """

#     model_name_or_path: str = field(
#         metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
#     )
#     config_name: Optional[str] = field(
#         default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
#     )
#     tokenizer_name: Optional[str] = field(
#         default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
#     )
#     cache_dir: Optional[str] = field(
#         default=None,
#         metadata={"help": "Path to directory to store the pretrained models downloaded from huggingface.co"},
#     )
#     model_revision: str = field(
#         default="main",
#         metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
#     )
#     use_auth_token: bool = field(
#         default=False,
#         metadata={
#             "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
#             "with private models)."
#         },
#     )


# @dataclass
# class DataTrainingArguments:
#     """
#     Arguments pertaining to what data we are going to input our model for training and eval.
#     """

#     dataset_name: Optional[str] = field(
#         default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
#     )
#     dataset_config_name: Optional[str] = field(
#         default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
#     )
#     train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
#     validation_file: Optional[str] = field(
#         default=None,
#         metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
#     )
#     test_file: Optional[str] = field(
#         default=None,
#         metadata={"help": "An optional input test data file to evaluate the perplexity on (a text file)."},
#     )
#     overwrite_cache: bool = field(
#         default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
#     )
#     preprocessing_num_workers: Optional[int] = field(
#         default=None,
#         metadata={"help": "The number of processes to use for the preprocessing."},
#     )
#     max_seq_length: int = field(
#         default=384,
#         metadata={
#             "help": "The maximum total input sequence length after tokenization. Sequences longer "
#             "than this will be truncated, sequences shorter will be padded."
#         },
#     )
#     pad_to_max_length: bool = field(
#         default=True,
#         metadata={
#             "help": "Whether to pad all samples to `max_seq_length`. "
#             "If False, will pad the samples dynamically when batching to the maximum length in the batch (which can "
#             "be faster on GPU but will be slower on TPU)."
#         },
#     )
#     max_train_samples: Optional[int] = field(
#         default=None,
#         metadata={
#             "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
#             "value if set."
#         },
#     )
#     max_eval_samples: Optional[int] = field(
#         default=None,
#         metadata={
#             "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
#             "value if set."
#         },
#     )
#     max_predict_samples: Optional[int] = field(
#         default=None,
#         metadata={
#             "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
#             "value if set."
#         },
#     )
#     version_2_with_negative: bool = field(
#         default=False, metadata={"help": "If true, some of the examples do not have an answer."}
#     )
#     null_score_diff_threshold: float = field(
#         default=0.0,
#         metadata={
#             "help": "The threshold used to select the null answer: if the best answer has a score that is less than "
#             "the score of the null answer minus this threshold, the null answer is selected for this example. "
#             "Only useful when `version_2_with_negative=True`."
#         },
#     )
#     doc_stride: int = field(
#         default=128,
#         metadata={"help": "When splitting up a long document into chunks, how much stride to take between chunks."},
#     )
#     n_best_size: int = field(
#         default=20,
#         metadata={"help": "The total number of n-best predictions to generate when looking for an answer."},
#     )
#     max_answer_length: int = field(
#         default=30,
#         metadata={
#             "help": "The maximum length of an answer that can be generated. This is needed because the start "
#             "and end predictions are not conditioned on one another."
#         },
#     )

#     def __post_init__(self):
#         if (
#             self.dataset_name is None
#             and self.train_file is None
#             and self.validation_file is None
#             and self.test_file is None
#         ):
#             raise ValueError("Need either a dataset name or a training/validation file/test_file.")
#         else:
#             if self.train_file is not None:
#                 extension = self.train_file.split(".")[-1]
#                 assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
#             if self.validation_file is not None:
#                 extension = self.validation_file.split(".")[-1]
#                 assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
#             if self.test_file is not None:
#                 extension = self.test_file.split(".")[-1]
#                 assert extension in ["csv", "json"], "`test_file` should be a csv or a json file."

def my_eval(args, eqn, accelerator, raw_datasets):

    print('calling my_eval in eval_for_classify_spans_hf.py (HuggingFace version)', file=sys.stderr)
    assert eqn['eqn_type'] == parse_eqn.eqn_types['classify_spans'], 'eval_for_classify_spans: eqn_type is classify, but eqn_type is: ' + str(
        parse_eqn.eqn_types_list[eqn['eqn_type']])

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    training_args = args['training_args']
    data_args = args['data_args']
    model_args = args['model_args']

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    local_rank = training_args.local_rank
    device = training_args.device
    n_gpu = training_args.n_gpu
    fp16 = training_args.fp16
    # output_dir = training_args.output_dir
    output_dir = None
    do_train = False
    do_eval = True
    do_predict = False # odd terminology; this is about applying the model to the test set, not to novel inputs
    overwrite_output_dir = True
    resume_from_checkpoint = True
    seed = training_args.seed
    max_seq_length = data_args.max_seq_length
    doc_stride = data_args.doc_stride

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {local_rank}, device: {device}, n_gpu: {n_gpu}"
        + f"distributed training: {bool(local_rank != -1)}, 16-bits training: {fp16}"
    )

    # logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    # last_checkpoint = None
    # if os.path.isdir(output_dir) and do_train and not overwrite_output_dir:
    #     last_checkpoint = get_last_checkpoint(output_dir)
    #     if last_checkpoint is None and len(os.listdir(output_dir)) > 0:
    #         raise ValueError(
    #             f"Output directory ({output_dir}) already exists and is not empty. "
    #             "Use --overwrite_output_dir to overcome."
    #         )
    #     elif last_checkpoint is not None and resume_from_checkpoint is None:
    #         logger.info(
    #             f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
    #             "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
    #         )

    # Set seed before initializing model.
    set_seed(seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    # if data_args.dataset_name is not None:
    #     # Downloading and loading a dataset from the hub.
    #     raw_datasets = load_dataset(
    #         data_args.dataset_name, data_args.dataset_config_name, cache_dir=model_args.cache_dir
    #     )
    # else:
    #     data_files = {}
    #     if data_args.train_file is not None:
    #         data_files["train"] = data_args.train_file
    #         extension = data_args.train_file.split(".")[-1]

    #     if data_args.validation_file is not None:
    #         data_files["validation"] = data_args.validation_file
    #         extension = data_args.validation_file.split(".")[-1]
    #     if data_args.test_file is not None:
    #         data_files["test"] = data_args.test_file
    #         extension = data_args.test_file.split(".")[-1]
    #     raw_datasets = load_dataset(extension, data_files=data_files, field="data", cache_dir=model_args.cache_dir)
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    
    # config = AutoConfig.from_pretrained(
    #     model_args.config_name if model_args.config_name else model_args.model_name_or_path,
    #     cache_dir=model_args.cache_dir,
    #     revision=model_args.model_revision,
    #     use_auth_token=True if model_args.use_auth_token else None,
    # )

    from gft.gft_internals.my_auto_model_hf import my_load_model_tokenizer_and_extractor
    model,tokenizer,extractor = my_load_model_tokenizer_and_extractor(args)

    # tokenizer = AutoTokenizer.from_pretrained(
    #     model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
    #     cache_dir=model_args.cache_dir,
    #     use_fast=True,
    #     revision=model_args.model_revision,
    #     use_auth_token=True if model_args.use_auth_token else None,
    # )
    # model = AutoModelForQuestionAnswering.from_pretrained(
    #     model_args.model_name_or_path,
    #     from_tf=bool(".ckpt" in model_args.model_name_or_path),
    #     config=config,
    #     cache_dir=model_args.cache_dir,
    #     revision=model_args.model_revision,
    #     use_auth_token=True if model_args.use_auth_token else None,
    # )

    # Tokenizer check: this script requires a fast tokenizer.
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError(
            "This example script only works for models that have a fast tokenizer. Checkout the big table of models "
            "at https://huggingface.co/transformers/index.html#supported-frameworks to find the model types that meet this "
            "requirement"
        )

    # Preprocessing the datasets.
    # Preprocessing is slighlty different for training and evaluation.

    if hasattr(raw_datasets["train"], 'column_names'):
        column_names = raw_datasets["train"].column_names
    else:
        column_names = raw_datasets["train"][0].keys()
    print('column_names: ' + str(column_names))

    x_field_names = eqn['x_field_names']
    y_field_names = eqn['y_field_names']
    assert len(y_field_names) == 1, 'classify_spans is currently limited to just one y variable [answers]: ' + str(y_field_names)
    assert len(x_field_names) == 2, 'classify_tokens is currently limited to exactly two x variables [questions, contexts]: ' + str(x_field_names)
    answer_column_name = y_field_names[0]
    question_column_name = x_field_names[0]
    context_column_name = x_field_names[1]

    # if do_train:
    #     column_names = raw_datasets["train"].column_names
    # elif do_eval:
    #     column_names = raw_datasets["validation"].column_names
    # else:
    #     column_names = raw_datasets["test"].column_names
    # question_column_name = "question" if "question" in column_names else column_names[0]
    # context_column_name = "context" if "context" in column_names else column_names[1]
    # answer_column_name = "answers" if "answers" in column_names else column_names[2]

    # Padding side determines if we do (question|context) or (context|question).
    pad_on_right = tokenizer.padding_side == "right"

    if max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(max_seq_length, tokenizer.model_max_length)

    # Training preprocessing
    def prepare_train_features(examples):
        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        examples[question_column_name] = [q.lstrip() for q in examples[question_column_name]]

        doc_stride = get_arg(args, 'doc_stride')
        pad_to_max_length = get_arg(args, 'pad_to_max_length')

        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if pad_to_max_length else False,
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        offset_mapping = tokenized_examples.pop("offset_mapping")

        # Let's label those examples!
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            answers = examples[answer_column_name][sample_index]
            # If no answers are given, set the cls_index as answer.
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Start/end character index of the answer in the text.
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                    token_end_index -= 1

                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

        return tokenized_examples

    max_train_samples = get_arg(args, 'max_train_samples')
    preprocessing_num_workers = get_arg(args, 'preprocessing_num_workers')
    overwrite_cache = get_arg(args, 'overwrite_cache')
    
    if do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if max_train_samples is not None:
            # We will select sample from whole data if argument is specified
            train_dataset = train_dataset.select(range(max_train_samples))
        # Create train feature from dataset
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                prepare_train_features,
                batched=True,
                num_proc=preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not overwrite_cache,
                desc="Running tokenizer on train dataset",
            )
        if max_train_samples is not None:
            # Number of samples might increase during Feature Creation, We select only specified max samples
            train_dataset = train_dataset.select(range(max_train_samples))

    # Validation preprocessing
    def prepare_validation_features(examples):
        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        examples[question_column_name] = [q.lstrip() for q in examples[question_column_name]]

        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if data_args.pad_to_max_length else False,
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
        # corresponding example_id and we will store the offset mappings.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        return tokenized_examples

    if do_eval:
        split = get_arg(args, 'split', 'test')
        if split not in raw_datasets:
            raise ValueError("cannot find split (%s) in dataset" % str(split))
        eval_examples = raw_datasets[split]
        if data_args.max_eval_samples is not None:
            # We will select sample from whole data
            eval_examples = eval_examples.select(range(data_args.max_eval_samples))
        # Validation Feature Creation
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_examples.map(
                prepare_validation_features,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )
        if data_args.max_eval_samples is not None:
            # During Feature creation dataset samples might increase, we will select required samples again
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

    if do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_examples = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            # We will select sample from whole data
            predict_examples = predict_examples.select(range(data_args.max_predict_samples))
        # Predict Feature Creation
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_examples.map(
                prepare_validation_features,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )
        if data_args.max_predict_samples is not None:
            # During Feature creation dataset samples might increase, we will select required samples again
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))

    # Data collator
    # We have already padded to max length if the corresponding flag is True, otherwise we need to pad in the data
    # collator.
    data_collator = (
        default_data_collator
        if data_args.pad_to_max_length
        else DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None)
    )

    # Post-processing:
    def post_processing_function(examples, features, predictions, stage="eval"):
        # Post-processing: we match the start logits and end logits to answers in the original context.
        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            version_2_with_negative=data_args.version_2_with_negative,
            n_best_size=data_args.n_best_size,
            max_answer_length=data_args.max_answer_length,
            null_score_diff_threshold=data_args.null_score_diff_threshold,
            output_dir=output_dir,   # used to be training_args.output_dir,
            log_level=log_level,
            prefix=stage,
        )
        # Format the result to the format the metric expects.
        # formatted_predictions = [
        #     {"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in predictions.items()
        # ]

        if data_args.version_2_with_negative:
            formatted_predictions = [
                {"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in predictions.items()
            ]
        else:
            formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]

        references = [{"id": ex["id"], "answers": ex[answer_column_name]} for ex in examples]
        return EvalPrediction(predictions=formatted_predictions, label_ids=references)

    metric_provider,metric_key = parse_metric_specification(args)    
    if metric_key is None:
        metric = load_metric("squad")
    else:
        print('load_metric: ' + metric_key, file=sys.stderr)
        metric = load_metric(*metric_key.split(','))

    # print('specified metric: ' + str(), file=sys.stderr)
    
    
    # if data_args.version_2_with_negative:
    #     print('loading metric: squad_v2', file=sys.stderr)
    #     metric = load_metric("squad_v2")
    # else:
    #     print('loading metric: squad', file=sys.stderr)
    #     metric = load_metric("squad")

    def compute_metrics(p: EvalPrediction):
        return metric.compute(predictions=p.predictions, references=p.label_ids)

    # Initialize our Trainer
    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if do_train else None,
        eval_dataset=eval_dataset if do_eval else None,
        eval_examples=eval_examples if do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=post_processing_function,
        compute_metrics=compute_metrics,
    )

    # Training
    if do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        model_provider,model_key = parse_model_specification(args)

        fig = get_arg(args, 'figure_of_merit', default=None)


        print('metrics: ' + str(metrics), file=sys.stderr)

        # metrics used to be eval_metric
        if fig is None:
            res = '\t'.join(['%s: %s' % (k, str(metrics[k])) for k in metrics])
        else:
            res = '%s: %s' % (fig, str(metrics[fig]))

        print('\t'.join(map(str, ['%0.2f seconds' % (time.time() - time0), model_key, str(res)])))

        # trainer.save_metrics("eval", metrics)

    # Prediction
    if do_predict:
        logger.info("*** Predict ***")
        results = trainer.predict(predict_dataset, predict_examples)
        metrics = results.metrics

        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

    # model_provider,model_key = parse_model_specification(args)
    # data_provider,data_key = parse_dataset_specification(args)

    # kwargs = {"finetuned_from": model_key, "tasks": "question-answering"}
    # if data_key is not None:
    #     kwargs["dataset_tags"] = data_key
    #     if hasattr(data_args, 'dataset_config_name') and data_args.dataset_config_name is not None:
    #         kwargs["dataset_args"] = data_args.dataset_config_name
    #         kwargs["dataset"] = f"{data_key} {data_args.dataset_config_name}"
    #     else:
    #         kwargs["dataset"] = data_key

    # if training_args.push_to_hub:
    #     trainer.push_to_hub(**kwargs)
    # else:
    #     trainer.create_model_card(**kwargs)


# def _mp_fn(index):
#     # For xla_spawn (TPUs)
#     main()


# if __name__ == "__main__":
#     main()


# # killroy

# MAX_GPU_BATCH_SIZE = 16
# EVAL_BATCH_SIZE = 16

# from accelerate import Accelerator, DistributedType

# from transformers import (
#     AdamW,
#     # AutoModelForSequenceClassification_hf,
#     # AutoTokenizer_hf,
#     get_linear_schedule_with_warmup,
#     set_seed,
#     AutoConfig,
#     DataCollatorWithPadding,
#     PretrainedConfig,
#     SchedulerType,
#     default_data_collator,
#     get_scheduler)

# from transformers import AutoModelForSequenceClassification as AutoModelForSequenceClassification_hf
# from transformers import AutoTokenizer as AutoTokenizer_hf


# logger = logging.getLogger(__name__)

# def my_eval(args, eqn, accelerator, raw_datasets, is_regression=False):

#     print('calling my_eval in eval_for_classify_hf.py (HuggingFace version)', file=sys.stderr)
#     if is_regression:
#         assert eqn['eqn_type'] == parse_eqn.eqn_types['regress'], 'eval_for_classify is for eqn_type: regress, but eqn_type is: ' + str(parse_eqn.eqn_types_list[eqn['eqn_type']])
#     else:
#         assert eqn['eqn_type'] == parse_eqn.eqn_types['classify'], 'eval_for_classify is for eqn_type: classify, but eqn_type is: ' + str(parse_eqn.eqn_types_list[eqn['eqn_type']])

#     # If passed along, set the training seed now.
#     seed = get_arg(args, 'seed', default=None)
#     print('seed: ' + str(seed), file=sys.stderr)
#     if not seed is None:
#         set_seed(seed)

#     # Sample hyper-parameters for learning rate, batch size, seed and a few other HPs
#     lr = get_arg(args, 'learning_rate', default=5e-5)
#     num_epochs = int(get_arg(args, 'num_train_epochs', default=15))
#     # correct_bias = config["correct_bias"]
#     batch_size = int(get_arg(args, 'per_device_train_batch_size', default=MAX_GPU_BATCH_SIZE))

#     dir = os.path.dirname(__file__)
#     if dir == '': dir = '.'

#     x_field_names = eqn['x_field_names']
#     y_field_names = eqn['y_field_names']

#     if not is_regression:
#         interned_labels,label_list = intern_labels(raw_datasets, y_field_names, args)
#         num_labels = len(interned_labels)
#     else:
#         interned_labels = label_list = None
#         num_labels = len(y_field_names)

#     model_provider,model_key = parse_model_specification(args, keyword='model')
#     base_model_provider,base_model_key = parse_model_specification(args, keyword='base_model')
#     assert model_provider == 'HuggingFace' or model_provider == 'Custom', 'Expected provider to be HuggingFace or Custom, but it was: ' + str(model_provider)

#     data_provider,data_key = parse_dataset_specification(args)
#     assert not (data_provider == 'PaddleHub' and is_regression), 'case not yet supported: regression on model_provider == HuggingFace and data_provider == PaddleHub'

#     print('model_key: ' + str(model_key), file=sys.stderr)

#     from my_auto_model_hf import my_load_model_tokenizer_and_extractor
#     model,tokenizer,extractor = my_load_model_tokenizer_and_extractor(args, keyword='model')

#     # model = AutoModelForSequenceClassification_hf.from_pretrained(model_key, return_dict=True, num_labels=num_labels)
#     # if not base_model_key is None: 
#     #     tokenizer = AutoTokenizer_hf.from_pretrained(base_model_key)
#     # else: 
#     #     tokenizer = AutoTokenizer_hf.from_pretrained(model_key)

#     metric_provider,metric_key = parse_metric_specification(args)

#     if metric_key is None:
#         if is_regression:
#             p = os.path.join(dir, 'sklearn_metrics/mean_squared_error.py')
#             print('load_metric: ' + p, file=sys.stderr)
#             metric = load_metric(p)
#             set_arg(args, 'better_figure_of_merit', -1) # less is more
#             if not get_arg(args, 'figure_of_merit', default=None): 
#                 set_arg(args, 'figure_of_merit', 'mean_squared_error')
#         else:
#             p = os.path.join(dir, 'sklearn_metrics/multiclass_glue.py')
#             print('load_metric: ' + p + ',mrpc', file=sys.stderr)
#             metric = load_metric(p,  'mrpc')
#     else:
#         print('load_metric: ' + metric_key, file=sys.stderr)
#         metric = load_metric(*metric_key.split(','))

#     device = accelerator.device

#     # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
#     accelerator = Accelerator()
#     # Make one log on every process with the configuration for debugging.
#     logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
#                         datefmt="%m/%d/%Y %H:%M:%S",
#                         level=logging.INFO)
#     logger.info(accelerator.state)

#     # Setup logging, we only want one process per machine to log things on the screen.
#     # accelerator.is_local_main_process is only True for one process per machine.
#     logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
#     if accelerator.is_local_main_process:
#         datasets.utils.logging.set_verbosity_warning()
#         transformers.utils.logging.set_verbosity_info()
#     else:
#         datasets.utils.logging.set_verbosity_error()
#         transformers.utils.logging.set_verbosity_error()

#     padding = "max_length" if get_arg(args, 'pad_to_max_length', default=False) else False

#     def preprocess_function(examples):

#         for y_field_name in y_field_names:
#             assert y_field_name in examples, 'tokenize_function: cannot find %s in %s' % (y_field_name, str(examples))
#         for x_field_name in x_field_names:
#             assert x_field_name in examples, 'tokenize_function: cannot find %s in %s' % (x_field_name, str(examples))

#         # Tokenize the texts
#         if len(x_field_names) == 1: texts = (examples[x_field_names[0]],)
#         else: texts = (examples[x_field_names[0]],examples[x_field_names[1]])

#         result = tokenizer(*texts, padding=padding, max_length=get_arg(args, 'max_length', default=128), truncation=True)

#         assert len(y_field_names) == 1, 'classify is currently limited to just one y variable: ' + str(y_field_names)
#         if interned_labels is None:
#             result['labels'] = [e for e in examples[y_field_names[0]]]
#         else:
#             ys = examples[y_field_names[0]]
#             if isinstance(ys, list):
#                 result['labels'] = [interned_labels[e] if e in interned_labels else e
#                                     for e in ys]
#             else:
#                 e = ys
#                 result['labels'] = interned_labels[e] if e in interned_labels else e
#         return result

#     try:
#         processed_datasets = raw_datasets.map(preprocess_function,
#                                               batched=True,
#                                               remove_columns=raw_datasets["train"].column_names,
#                                               desc="Running tokenizer on dataset",)
#     except:
#         print('warning could not remove_columns=raw_datasets["train"].column_names', file=sys.stderr)
#         # processed_datasets = raw_datasets.map(preprocess_function,
#         #                                       batched=True,
#         #                                       desc="Running tokenizer on dataset",)
#         assert 'val' in raw_datasets, 'cannot find validation set'
#         processed_datasets = { 'train' : [preprocess_function(e) for e in raw_datasets['train']],
#                                'val' : [preprocess_function(e) for e in raw_datasets['val']]}

#     # assert 'val' in processed_datasets, 'cannot find validation set'
#     train_dataset = processed_datasets['train']
#     eval_dataset = processed_datasets[get_arg(args, 'split', default='val')]

#     # Log a few random samples from the training set:
#     # for index in random.sample(range(len(train_dataset)), 3):
#     #     logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

#     # DataLoaders creation:
#     if get_arg(args, 'pad_to_max_length', default=False):
#         # If padding was already done ot max length, we use the default data collator that will just convert everything
#         # to tensors.
#         data_collator = default_data_collator
#     else:
#         # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
#         # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
#         # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
#         data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None))

#     bs = get_arg(args, 'per_device_train_batch_size', default=MAX_GPU_BATCH_SIZE)
#     ebs = get_arg(args, 'per_device_eval_batch_size', default=EVAL_BATCH_SIZE)
#     train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=bs)
#     eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=bs)

#     # Optimizer
#     # Split weights in two groups, one with weight decay and the other not.
#     no_decay = ["bias", "LayerNorm.weight"]
#     optimizer_grouped_parameters = [
#         {
#             "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
#             "weight_decay": get_arg(args, 'weight_decay', default=0.0),
#         },
#         {
#             "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
#             "weight_decay": 0.0,
#         },
#     ]
#     optimizer = AdamW(optimizer_grouped_parameters, lr=lr)

#     # Prepare everything with our `accelerator`.
#     model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(model, optimizer, train_dataloader, eval_dataloader)

#     # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
#     # shorter in multiprocess)

#     # Scheduler and math around the number of training steps.
#     gas = get_arg(args, 'gradient_accumulation_steps', default=1)
#     num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gas)
#     epochs = int(get_arg(args, 'num_train_epochs', default=15))
#     max_train_steps = get_arg(args, 'max_train_steps', default=None)
#     if max_train_steps is None:
#         max_train_steps = epochs * num_update_steps_per_epoch
#     else:
#         epochs = int(math.ceil(max_train_steps / num_update_steps_per_epoch))

#     lr_scheduler = get_scheduler(
#         name=get_arg(args, 'lr_scheduler_type', default="linear"),
#         optimizer=optimizer,
#         num_warmup_steps=get_arg(args, 'num_warmup_steps', default=0),
#         num_training_steps=max_train_steps,
#     )

#     # Train!

#     # total_batch_size = bs * accelerator.num_processes * gas

#     # logger.info("***** Running training *****")
#     # logger.info(f"  Num examples = {len(train_dataset)}")
#     # logger.info(f"  Num Epochs = {epochs}")
#     # logger.info(f"  Instantaneous batch size per device = {bs}")
#     # logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
#     # logger.info(f"  Gradient Accumulation steps = {gas}")
#     # logger.info(f"  Total optimization steps = {max_train_steps}")
#     # completed_steps = 0

#     # best_so_far = None
#     time0 = time.time()

#     model.eval()
#     for step, batch in enumerate(eval_dataloader):
#         outputs = model(**batch)
#         predictions = outputs.logits.argmax(dim=-1) if not is_regression else outputs.logits.squeeze()
#         metric.add_batch(predictions=accelerator.gather(predictions),
#                          references=accelerator.gather(batch["labels"]))
        
#     eval_metric = metric.compute()
#     logger.info(f"{eval_metric} {time.time() - time0} seconds")
#     fig = get_arg(args, 'figure_of_merit', default=None)

#     if fig is None:
#         res = '\t'.join(['%s: %s' % (k, str(eval_metric[k])) for k in eval_metric])
#     else:
#         res = '%s: %s' % (fig, str(eval_metric[fig]))

#     accelerator.print('\t'.join(map(str, ['%0.2f seconds' % (time.time() - time0), model_key, res])))
    
