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
Fine-tuning a 🤗 Transformers model on span tasks (SQuAD) relying on the accelerate library without using a Trainer.
"""

# based on https://github.com/huggingface/transformers/blob/master/examples/pytorch/question-answering/run_qa_no_trainer.py

# convention: _hf is from HuggingFace, and _pd is from PaddleHub

# import pdb
import gft
import torch,time
import numpy as np
from functools import partial

import sys,os,shutil
import seqeval,time,sys
import numpy as np

print(__name__ + ': loading from paddle', file=sys.stderr)

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
from gft.gft_internals.gft_util import intern_labels, parse_model_specification, parse_dataset_specification, parse_metric_specification, better, checkpoint_filename, get_arg, set_arg

import transformers
from accelerate import Accelerator

from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AdamW,
    AutoConfig,
    # AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    set_seed,
    DataCollatorWithPadding,
    # AutoModelForQuestionAnswering,
    EvalPrediction,
)

import paddle
from paddle.io import DataLoader as DataLoader_pd
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.transformers import (
    AutoModelForQuestionAnswering as AutoModelForQuestionAnswering_pd,
    AutoTokenizer as AutoTokenizer_pd,
)
from paddlenlp.transformers import LinearDecayWithWarmup

# from transformers.file_utils import get_full_repo_name
# from transformers.utils import check_min_version
# from transformers.utils.versions import require_version
from .utils_qa import postprocess_qa_predictions


logger = logging.getLogger(__name__)

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.17.0.dev0")
# require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/question-answering/requirements.txt")

# You should update this to your particular problem to have better documentation of `model_type`
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.17.0.dev0")

# require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/question-answering/requirements.txt") 

logger = logging.getLogger(__name__)
# You should update this to your particular problem to have better documentation of `model_type`
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def fit(args, eqn, accelerator, raw_datasets):
    print('calling fit in fit_for_classify_spans.py')
    assert eqn['eqn_type'] == parse_eqn.eqn_types['classify_spans'], \
        'fit_for_classify_tokens is for eqn_type: classify_spans, but eqn_type is: ' + str(eqn['eqn_type'])

    # If passed along, set the training seed now.
    seed = get_arg(args, 'seed')
    print('seed: ' + str(seed))
    if not seed is None:
        set_seed(seed)

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
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

    # Handle the repository creation
    output_dir = get_arg(args, 'output_dir', default=None)

    if accelerator.is_main_process:
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
        accelerator.wait_for_everyone()

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    column_names = raw_datasets["train"][0].keys()
    print('column_names: ' + str(column_names))

    x_field_names = eqn['x_field_names']
    y_field_names = eqn['y_field_names']
    assert len(y_field_names) == 1, 'classify_spans is currently limited to just one y variable [answers]: ' + str(y_field_names)
    assert len(x_field_names) == 2, 'classify_tokens is currently limited to exactly two x variables [questions, contexts]: ' + str(x_field_names)
    answer_column_name = y_field_names[0]
    question_column_name = x_field_names[0]
    context_column_name = x_field_names[1]

    num_labels = 2
    label_to_id = {i:i for i in range(num_labels)}

    model_provider, model_key = parse_model_specification(args)
    assert not model_provider == 'PaddlePaddle', 'PaddlePaddle models are not supported (yet)'  # stub to flesh out later
    tokenizer = AutoTokenizer_pd.from_pretrained(model_key)
    model = AutoModelForQuestionAnswering_pd.from_pretrained(model_key)
    max_seq_length = tokenizer.model_max_length

    # max_seq_length -= 5         # gross hack to avoid crashing

    print('max_seq_length = ' + str(max_seq_length))

    model.resize_token_embeddings(len(tokenizer))

    # Padding side determines if we do (question|context) or (context|question).
    pad_on_right = tokenizer.padding_side == "right"

    pad_to_max_length = get_arg(args, 'pad_to_max_length', default=False)
    doc_stride = get_arg(args, 'doc_stride')


    # Training preprocessing
    def prepare_train_features(examples):
        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace

        print('entering prepare_train_features: max_seq_length = %s, doc_stride = %s, pad_to_max_length = %s' % (str(max_seq_length), str(doc_stride), str(pad_to_max_length)), file=sys.stderr)

        examples[question_column_name] = [q.lstrip() for q in examples[question_column_name]]

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

        stok = str(tokenized_examples)
        l = len(stok)
        if l > 150: stok = stok[0:150]
        print('leaving prepare_train_features (len: %d): %s' % (l, stok), file=sys.stderr)
        return tokenized_examples

    # killroy

    max_train_samples = get_arg(args, 'max_train_samples')

    if "train" not in raw_datasets:
        raise ValueError("--do_train requires a train dataset")
    train_dataset = raw_datasets["train"]
    if max_train_samples is not None:
        # We will select sample from whole data if agument is specified
        train_dataset = train_dataset.select(range(max_train_samples))

    # Create train feature from dataset
    with accelerator.main_process_first():    

        print('batched: %s, num_proc: %s, remove_columns: %s, load_from_cache_file: %s' % (str(True), str(get_arg(args, 'preprocessing_num_workers')), str(column_names), str(False)), file=sys.stderr)

        data_provider, _ = parse_dataset_specification(args)
        if data_provider != "PaddleHub":
            train_dataset = train_dataset.map(
                prepare_train_features,
                batched=True,
                num_proc=get_arg(args, 'preprocessing_num_workers'),
                remove_columns=column_names,
                load_from_cache_file=False,
                desc="Running tokenizer on train dataset (batched = %s, num_proc = %s) % (str(batched), str(num_proc))",
            )
        else:
            num_workers = int(get_arg(args, 'preprocessing_num_workers', default=20))
            num_workers = 1     # for debugging
            train_dataset = train_dataset.map(
                prepare_train_features,
                batched=True,
                num_workers=num_workers,
                lazy=True,
            )

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
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if pad_to_max_length else False,
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

    if "val" not in raw_datasets:
        raise ValueError("--do_eval requires a val dataset")
    eval_examples = raw_datasets["val"]
    max_eval_samples = get_arg(args, 'max_eval_samples')
    if max_eval_samples is not None:
        # We will select sample from whole data
        eval_examples = eval_examples.select(range(max_eval_samples))
    # Validation Feature Creation

    with accelerator.main_process_first():

        data_provider, _ = parse_dataset_specification(args)
        if data_provider != "PaddleHub":
            eval_dataset = eval_examples.map(
                prepare_validation_features,
                batched=True,
                num_proc=get_arg(args, 'preprocessing_num_workers'),
                remove_columns=column_names,
                load_from_cache_file=False,
                desc="Running tokenizer on validation dataset",
            )
        else:
            eval_dataset = eval_examples.map(
                prepare_validation_features,
                batched=True,
                num_workers=int(get_arg(args, 'preprocessing_num_workers', default=20)),
                lazy=True,
            )

    if max_eval_samples is not None:
        # During Feature creation dataset samples might increase, we will select required samples again
        eval_dataset = eval_dataset.select(range(max_eval_samples))

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    if pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None))

    train_dataloader = DataLoader_pd(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=get_arg(args, 'per_device_train_batch_size')
    )

    eval_dataset_for_model = eval_dataset.remove_columns(["example_id", "offset_mapping"])
    eval_dataloader = DataLoader_pd(
        eval_dataset_for_model, collate_fn=data_collator, batch_size=get_arg(args, 'per_device_eval_batch_size')
    )

    metric_provider, metric_key = parse_metric_specification(args)
    assert not metric_provider == 'PaddlePaddle', 'PaddlePaddle metrics are not supported (yet)'  # stub to flesh out later


    if metric_key is None:
        print('load_metric: squad')
        val_metric = load_metric("squad")
        train_metric = load_metric("squad")
    else:
        print('load_metric: ' + metric_key)
        val_metric = load_metric(*metric_key.split(','))
        train_metric = load_metric(*metric_key.split(','))

    # Post-processing:
    def post_processing_function(examples, features, predictions, stage="eval"):
        # Post-processing: we match the start logits and end logits to answers in the original context.
        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            version_2_with_negative=(metric_key == 'squad_v2'),
            n_best_size=int(get_arg(args, 'n_best_size', default=20)),
            max_answer_length=int(get_arg(args, 'max_answer_length', default=30)),
            null_score_diff_threshold=float(get_arg(args, 'null_score_diff_threshold', default=0.0)),
            output_dir=output_dir,
            prefix=stage,
        )

        # Format the result to the format the metric expects.
        formatted_predictions = [{"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in predictions.items()]
        references = [{"id": ex["id"], "answers": ex[answer_column_name]} for ex in examples]
        return EvalPrediction(predictions=formatted_predictions, label_ids=references)

    # metric = load_metric("squad_v2" if get_arg(args, 'version_2_with_negative') else "squad")

    # Create and fill numpy array of size len_of_validation_data * max_length_of_output_tensor
    def create_and_fill_np_array(start_or_end_logits, dataset, max_len):
        """
        Create and fill numpy array of size len_of_validation_data * max_length_of_output_tensor

        Args:
            start_or_end_logits(:obj:`tensor`):
                This is the output predictions of the model. We can only enter either start or end logits.
            eval_dataset: Evaluation dataset
            max_len(:obj:`int`):
                The maximum length of the output tensor. ( See the model.eval() part for more details )
        """

        step = 0
        # create a numpy array and fill it with -100.
        logits_concat = np.full((len(dataset), max_len), -100, dtype=np.float64)
        # Now since we have create an array now we will populate it with the outputs gathered using accelerator.gather
        for i, output_logit in enumerate(start_or_end_logits):  # populate columns
            # We have to fill it such that we have to take the whole tensor and replace it on the newly created array
            # And after every iteration we have to change the step

            batch_size = output_logit.shape[0]
            cols = output_logit.shape[1]

            if step + batch_size < len(dataset):
                logits_concat[step : step + batch_size, :cols] = output_logit
            else:
                logits_concat[step:, :cols] = output_logit[: len(dataset) - step]

            step += batch_size

        return logits_concat

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": get_arg(args, 'weight_decay'),
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=get_arg(args, 'learning_rate'))

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
    # shorter in multiprocess)

    # Scheduler and math around the number of training steps.
    gas = get_arg(args, 'gradient_accumulation_steps', default=1)
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gas)
    max_train_steps = get_arg(args, 'max_train_steps', default=None)
    epochs = int(get_arg(args, 'num_train_epochs', default=15))

    if max_train_steps is None:
        max_train_steps = epochs * num_update_steps_per_epoch
    else:
        epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name=get_arg(args, 'lr_scheduler_type', default="linear"),
        optimizer=optimizer,
        num_warmup_steps=get_arg(args, 'num_warmup_steps', default=0),
        num_training_steps=max_train_steps,
    )

    # Train!
    bs = get_arg(args, 'per_device_train_batch_size', default=32)
    eval_bs = get_arg(args, 'per_device_eval_batch_size', default=32)
    total_batch_size = bs * accelerator.num_processes * gas

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {epochs}")
    logger.info(f"  Instantaneous batch size per device = {bs}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {gas}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0

    best_so_far = None
    time0 = time.time()

    # for epoch in range(epochs):
    #     model.train()
    #     for step, batch in enumerate(train_dataloader):
    #         outputs = model(**batch)
    #         loss = outputs.loss
    #         loss = loss / gas
    #         accelerator.backward(loss)
    #         if step % gas == 0 or step == len(train_dataloader) - 1:
    #             optimizer.step()
    #             lr_scheduler.step()
    #             optimizer.zero_grad()
    #             # progress_bar.update(1)
    #             completed_steps += 1

    #         if completed_steps >= max_train_steps:
    #             break

        # # Evaluation
        # logger.info("***** Running Evaluation *****")
        # logger.info(f"  Num examples = {len(eval_examples)}")
        # logger.info(f"  Batch size = {eval_bs}")

        # all_start_logits = []
        # all_end_logits = []
        # for step, batch in enumerate(eval_dataloader):
        #     with torch.no_grad():
        #         outputs = model(**batch)
        #         start_logits = outputs.start_logits
        #         end_logits = outputs.end_logits

        #     if pad_to_max_length:  # necessary to pad predictions and labels for being gathered
        #         start_logits = accelerator.pad_across_processes(start_logits, dim=1, pad_index=-100)
        #         end_logits = accelerator.pad_across_processes(end_logits, dim=1, pad_index=-100)

        #     all_start_logits.append(accelerator.gather(start_logits).cpu().numpy())
        #     all_end_logits.append(accelerator.gather(end_logits).cpu().numpy())

        #     max_len = max([x.shape[1] for x in all_start_logits])  # Get the max_length of the tensor

        #     # concatenate the numpy array
        #     start_logits_concat = create_and_fill_np_array(all_start_logits, eval_examples, max_len)
        #     end_logits_concat = create_and_fill_np_array(all_end_logits, eval_examples, max_len)

        #     # delete the list of numpy arrays
        #     del all_start_logits
        #     del all_end_logits

        #     outputs_numpy = (start_logits_concat, end_logits_concat)
        #     prediction = post_processing_function(eval_examples, eval_dataset, outputs_numpy)
        #     eval_metric = metric.compute(predictions=prediction.predictions, references=prediction.label_ids)
        #     logger.info(f"Evaluation metrics: {eval_metric}")


        # fn = checkpoint_filename(args, model_key, epoch, False)
        # b,best = better(args, eval_metric, best_so_far)
        # if b:
        #     best_so_far = best
        #     print('best_so_far: %f' % (best_so_far))
        #     fn = checkpoint_filename(args, model_key, epoch, True)
        # if not output_dir is None:
        #     prev = checkpoint_filename(args, model_key, epoch-1, False)
        #     try:
        #         if os.path.exists(prev): shutil.rmtree(prev)
        #     except:
        #         print('failed to delete: ' + str(prev), sys.stderr)
        #     model.save_pretrained(fn)


    for epoch in range(epochs):
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
                progress_bar.update(1)
                completed_steps += 1

            if completed_steps >= max_train_steps:
                break

        if epoch < epochs - 1:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
            if accelerator.is_main_process:
                tokenizer.save_pretrained(output_dir)
                # repo.push_to_hub(
                #     commit_message=f"Training in progress epoch {epoch}", blocking=False, auto_lfs_prune=True
                # )

    # Evaluation
    logger.info("***** Running Evaluation *****")
    logger.info(f"  Num examples = {len(eval_dataset)}")
    logger.info(f"  Batch size = {eval_bs}")

    all_start_logits = []
    all_end_logits = []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(**batch)
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits

            if not pad_to_max_length:  # necessary to pad predictions and labels for being gathered
                start_logits = accelerator.pad_across_processes(start_logits, dim=1, pad_index=-100)
                end_logits = accelerator.pad_across_processes(end_logits, dim=1, pad_index=-100)

            all_start_logits.append(accelerator.gather(start_logits).cpu().numpy())
            all_end_logits.append(accelerator.gather(end_logits).cpu().numpy())

    max_len = max([x.shape[1] for x in all_start_logits])  # Get the max_length of the tensor

    # concatenate the numpy array
    start_logits_concat = create_and_fill_np_array(all_start_logits, eval_dataset, max_len)
    end_logits_concat = create_and_fill_np_array(all_end_logits, eval_dataset, max_len)

    # delete the list of numpy arrays
    del all_start_logits
    del all_end_logits

    outputs_numpy = (start_logits_concat, end_logits_concat)
    prediction = post_processing_function(eval_examples, eval_dataset, outputs_numpy)
    eval_metric = val_metric.compute(predictions=prediction.predictions, references=prediction.label_ids)
    logger.info(f"Evaluation metrics: {eval_metric}")

    # Prediction
    # if False
    #     logger.info("***** Running Prediction *****")
    #     logger.info(f"  Num examples = {len(predict_dataset)}")
    #     logger.info(f"  Batch size = {args.per_device_eval_batch_size}")

    #     all_start_logits = []
    #     all_end_logits = []
    #     for step, batch in enumerate(predict_dataloader):
    #         with torch.no_grad():
    #             outputs = model(**batch)
    #             start_logits = outputs.start_logits
    #             end_logits = outputs.end_logits

    #             if not args.pad_to_max_length:  # necessary to pad predictions and labels for being gathered
    #                 start_logits = accelerator.pad_across_processes(start_logits, dim=1, pad_index=-100)
    #                 end_logits = accelerator.pad_across_processes(start_logits, dim=1, pad_index=-100)

    #             all_start_logits.append(accelerator.gather(start_logits).cpu().numpy())
    #             all_end_logits.append(accelerator.gather(end_logits).cpu().numpy())

    #     max_len = max([x.shape[1] for x in all_start_logits])  # Get the max_length of the tensor
    #     # concatenate the numpy array
    #     start_logits_concat = create_and_fill_np_array(all_start_logits, predict_dataset, max_len)
    #     end_logits_concat = create_and_fill_np_array(all_end_logits, predict_dataset, max_len)

    #     # delete the list of numpy arrays
    #     del all_start_logits
    #     del all_end_logits

    #     outputs_numpy = (start_logits_concat, end_logits_concat)
    #     prediction = post_processing_function(predict_examples, predict_dataset, outputs_numpy)
    #     predict_metric = metric.compute(predictions=prediction.predictions, references=prediction.label_ids)
    #     logger.info(f"Predict metrics: {predict_metric}")

    if output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)
            # if args.push_to_hub:
            #     repo.push_to_hub(commit_message="End of training", auto_lfs_prune=True)


# if __name__ == "__main__":
#     main()
