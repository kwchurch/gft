# based on https://huggingface.co/blog/fine-tune-wav2vec2-english
# and https://github.com/huggingface/transformers/tree/master/examples/pytorch/speech-recognition


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

""" Fine-tuning a 🤗 Transformers CTC model for automatic speech recognition"""
import gft
import functools
import json
import logging
import os
import re
import sys
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import datasets
import numpy as np
import torch
from datasets import DatasetDict, load_dataset, load_metric

import transformers
from transformers import (
    # AutoConfig,
    # AutoFeatureExtractor,
    # AutoModelForCTC,
    # AutoProcessor,
    # AutoTokenizer,
    HfArgumentParser,
    # Trainer,
    TrainingArguments,
    # Wav2Vec2Processor,
    # set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

def list_field(default=None, metadata=None):
    return field(default_factory=lambda: default, metadata=metadata)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    # formally model_name_or_path
    model: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models (replaces model_name_or_path)."}
    )

    base_model: str = field(
        default=None,
        metadata={"help": "prefix (H/P/C): base model | checkpoint | adapter"},
        )

    do_not_catch_errors: bool = field(
        default=False,
        metadata={"help": "Do not catch errors if True [default = False]"},
    )

    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pretrained tokenizer or tokenizer identifier from huggingface.co/models"},
    )

    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )

    freeze_feature_encoder: bool = field(
        default=True, metadata={"help": "Whether to freeze the feature encoder layers of the model."}
    )

    attention_dropout: float = field(
        default=0.0, metadata={"help": "The dropout ratio for the attention probabilities."}
    )

    activation_dropout: float = field(
        default=0.0, metadata={"help": "The dropout ratio for activations inside the fully connected layer."}
    )

    feat_proj_dropout: float = field(default=0.0, metadata={"help": "The dropout ratio for the projected features."})

    hidden_dropout: float = field(
        default=0.0,
        metadata={
            "help": "The dropout probability for all fully connected layers in the embeddings, encoder, and pooler."
        },
    )

    final_dropout: float = field(
        default=0.0,
        metadata={"help": "The dropout probability for the final projection layer."},
    )

    mask_time_prob: float = field(
        default=0.05,
        metadata={
            "help": "Probability of each feature vector along the time axis to be chosen as the start of the vector"
            "span to be masked. Approximately ``mask_time_prob * sequence_length // mask_time_length`` feature"
            "vectors will be masked along the time axis."
        },
    )

    mask_time_length: int = field(
        default=10,
        metadata={"help": "Length of vector span to mask along the time axis."},
    )

    mask_feature_prob: float = field(
        default=0.0,
        metadata={
            "help": "Probability of each feature vector along the feature axis to be chosen as the start of the vector"
            "span to be masked. Approximately ``mask_feature_prob * sequence_length // mask_feature_length`` feature bins will be masked along the time axis."
        },
    )

    mask_feature_length: int = field(
        default=10,
        metadata={"help": "Length of vector span to mask along the feature axis."},
    )

    layerdrop: float = field(default=0.0, metadata={"help": "The LayerDrop probability."})

    ctc_loss_reduction: Optional[str] = field(
        default="mean", metadata={"help": "The way the ctc loss should be reduced. Should be one of 'mean' or 'sum'."}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    # formally dataset_name

    data: str = field(
        default=None,
        metadata={"help": "A dataset from HuggingFace (https://huggingface.co/datasets), PaddlePaddle or a custom dataset (--csv_dataset) -- may be prefaced by H:, P: or C: to disambiguate.\n\
If there are two comma separated values, then the first is the name of a dataset and the second is the name of the config (replaces dataset_name and datase_config_name)."}
    )

    dataset_config_name: str = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )

    splits: str = field(
        default=None, metadata={"help": "3 comma separate values for train,val,test (if dataset does not have a val set, use __select_from_train__ for val) (replaces train_split_name and eval_split_name)."}
    )

    split: str = field(
        default='test', 
        metadata={"help": "train,val,test, etc."}
    )

    task: str = field(
        default=None,
        metadata={"help": "see https://huggingface.co/docs/transformers/v4.16.2/en/main_classes/pipelines#transformers.pipeline.task"}
    )

    # deprecated

    # train_split_name: str = field(
    #     default="train+validation",
    #     metadata={
    #         "help": "The name of the training data set split to use (via the datasets library). Defaults to 'train'"
    #     },
    # )
    # eval_split_name: str = field(
    #     default="test",
    #     metadata={
    #         "help": "The name of the training data set split to use (via the datasets library). Defaults to 'train'"
    #     },
    # )


    # parser.add_argument("-e", "--eqn", type=str,
    #     help="classify: y ~ text1 + text2, \n\
    #                     where the variables are the names of fields in csv files (spaces are not allowed in field names).\n\
    #                     Here are some more examples:\n\
    #                     regress: y ~ text1 + text2; \n\
    #                     regress: Valence + Arousal + Dominance ~ word",
    #     required=True)

    eqn: str = field(default=None, 
                     metadata={"help": "classify: y ~ text1 + text2, \n\
                     where the variables are the names of fields in csv files (spaces are not allowed in field names).\n\
                     Here are some more examples:\n\
                     regress: y ~ text1 + text2; \n\
                     regress: Valence + Arousal + Dominance ~ word;\
                     (replaces audio_column_name and text_column_name)."}
    )

    metric: str = field(default=None, metadata={"help": "examples: H:accuracy, C:<filename to a custom metric>; <provider string>:key, where provider string is HuggingFace (H), PaddlePaddle (P) or Custome (C); See https://huggingface.co/docs/datasets/using_metrics.html for HuggingFace metrics"})

    figure_of_merit: str = field(default=None,
                                 metadata={"help": "defaults to accuracy for classification or mean_squared_error for regression; should be a value returned from --HuggingFace_metric"})

    better_figure_of_merit: int = field(default=1, metadata={"help": "use -1 if metric prefers smaller values"})

    data_dir: str = field(default=None, metadata={"help": "optional argument to HuggingFace datasets.load_dataset (usually not needed)"})

    max_seq_length: int = field(default=384, metadata={"help": "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded if `--pad_to_max_lengh` is passed."})

    pad_to_max_length: bool = field(default=False, metadata= {"help": "If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used."})
    max_length: int = field(default=128, metadata = {"help": "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
                                                     " sequences shorter will be padded if `--pad_to_max_length` is passed."},)

    doc_stride: int = field(default=128, metadata = {"help": "When splitting up a long document into chunks how much stride to take between chunks."},)

    version_2_with_negative: bool = field(
        default=False, metadata={"help": "If true, some of the examples do not have an answer."}
    )
    null_score_diff_threshold: float = field(
        default=0.0,
        metadata={
            "help": "The threshold used to select the null answer: if the best answer has a score that is less than "
            "the score of the null answer minus this threshold, the null answer is selected for this example. "
            "Only useful when `version_2_with_negative=True`."
        },
    )

    n_best_size: int = field(default=20, metadata = {"help": "The total number of n-best predictions to generate when looking for an answer."},)
    max_duration_in_seconds: float = field(default=20.0, metadata={"help": "Filter audio files that are longer than `max_duration_in_seconds` seconds to 'max_duration_in_seconds`"},)
    min_duration_in_seconds: float = field(default=0.0, metadata={"help": "Filter audio files that are shorter than `min_duration_in_seconds` seconds"},)
    is_split_into_words: bool = field(default=False, metadata = {"help": "use for datasets that are already split into words."},)

    # deprecated

    # audio_column_name: str = field(
    #     default="audio",
    #     metadata={"help": "The name of the dataset column containing the audio data. Defaults to 'audio'"},
    # )
    # text_column_name: str = field(
    #     default="text",
    #     metadata={"help": "The name of the dataset column containing the text data. Defaults to 'text'"},
    # )

    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )

    preprocessing_num_workers: Optional[int] = field(
        # -svail-
        # default=None,
        default=20,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
            "value if set."
        },
    )

    # -svail-
    # -srun fail when parse list_field, change to a string and process later
    # chars_to_ignore: Optional[List[str]] = list_field(
    #     default=None,
    #     metadata={"help": "A list of characters to remove from the transcripts."},
    # )

    chars_to_ignore: Optional[str] = field(
        default=",?.!-\;\:\"“%‘”�",
        metadata={"help": "A list of characters to remove from the transcripts."},)


    preprocessing_only: bool = field(
        default=False,
        metadata={
            "help": "Whether to only do data preprocessing and skip training. "
            "This is especially useful when data preprocessing errors out in distributed training due to timeout. "
            "In this case, one should run the preprocessing in a non-distributed setup with `preprocessing_only=True` "
            "so that the cached datasets can consequently be loaded in distributed training"
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "If :obj:`True`, will use the token generated when running"
            ":obj:`transformers-cli login` as HTTP bearer authorization for remote files."
        },
    )
    unk_token: str = field(default="[UNK]", metadata={"help": "The unk token for the tokenizer"})
    pad_token: str = field(default="[PAD]", metadata={"help": "The padding token for the tokenizer"})
    word_delimiter_token: str = field(default="|", metadata={"help": "The word delimiter token for the tokenizer"})

    # for ctc
    phoneme_language: Optional[str] = field(
        default=None,
        metadata={
            "help": "The target language that should be used be"
            " passed to the tokenizer for tokenization. Note that"
            " this is only relevant if the model classifies the"
            " input audio to a sequence of phoneme sequences."
        })

    # for classify_spans
    max_answer_length: int = field(
        default=30,
        metadata={
            "help": "The maximum length of an answer that can be generated. This is needed because the start "
            "and end predictions are not conditioned on one another."
        },)


# @dataclass
# class DataCollatorCTCWithPadding:
#     """
#     Data collator that will dynamically pad the inputs received.
#     Args:
#         processor (:class:`~transformers.AutoProcessor`)
#             The processor used for proccessing the data.
#         padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
#             Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
#             among:
#             * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
#               sequence if provided).
#             * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
#               maximum acceptable input length for the model if that argument is not provided.
#             * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
#               different lengths).
#         max_length (:obj:`int`, `optional`):
#             Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
#         max_length_labels (:obj:`int`, `optional`):
#             Maximum length of the ``labels`` returned list and optionally padding length (see above).
#         pad_to_multiple_of (:obj:`int`, `optional`):
#             If set will pad the sequence to a multiple of the provided value.
#             This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
#             7.5 (Volta).
#     """

#     processor: AutoProcessor
#     padding: Union[bool, str] = "longest"
#     pad_to_multiple_of: Optional[int] = None
#     pad_to_multiple_of_labels: Optional[int] = None

#     def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
#         # split inputs and labels since they have to be of different lenghts and need
#         # different padding methods
#         input_features = [{"input_values": feature["input_values"]} for feature in features]
#         label_features = [{"input_ids": feature["labels"]} for feature in features]

#         batch = self.processor.pad(
#             input_features,
#             padding=self.padding,
#             pad_to_multiple_of=self.pad_to_multiple_of,
#             return_tensors="pt",
#         )

#         with self.processor.as_target_processor():
#             labels_batch = self.processor.pad(
#                 label_features,
#                 padding=self.padding,
#                 pad_to_multiple_of=self.pad_to_multiple_of_labels,
#                 return_tensors="pt",
#             )

#         # replace padding with -100 to ignore loss correctly
#         labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

#         batch["labels"] = labels

#         return batch


def create_vocabulary_from_data(
    datasets: DatasetDict,
    word_delimiter_token: Optional[str] = None,
    unk_token: Optional[str] = None,
    pad_token: Optional[str] = None,
):
    # Given training and test labels create vocabulary
    def extract_all_chars(batch):
        all_text = " ".join(batch["target_text"])
        vocab = list(set(all_text))
        return {"vocab": [vocab], "all_text": [all_text]}

    vocabs = datasets.map(
        extract_all_chars,
        batched=True,
        batch_size=-1,
        keep_in_memory=True,
        remove_columns=datasets["train"].column_names,
    )

    # take union of all unique characters in each dataset
    vocab_set = functools.reduce(
        lambda vocab_1, vocab_2: set(vocab_1["vocab"][0]) | set(vocab_2["vocab"][0]), vocabs.values()
    )

    vocab_dict = {v: k for k, v in enumerate(sorted(list(vocab_set)))}

    # replace white space with delimiter token
    if word_delimiter_token is not None:
        vocab_dict[word_delimiter_token] = vocab_dict[" "]
        del vocab_dict[" "]

    # add unk and pad token
    if unk_token is not None:
        vocab_dict[unk_token] = len(vocab_dict)

    if pad_token is not None:
        vocab_dict[pad_token] = len(vocab_dict)

    return vocab_dict

def gft_parse_args():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    return { "model_args": model_args, 
             "data_args": data_args, 
             "training_args" : training_args }
