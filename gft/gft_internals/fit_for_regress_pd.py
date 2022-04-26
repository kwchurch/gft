
# based on https://github.com/huggingface/transformers/blob/master/examples/pytorch/text-classification/run_glue_no_trainer.py

# convention: _hf is from HuggingFace, and _pd is from PaddleHub

# import pdb
import gft
import sys,os,shutil,argparse,torch,time
import numpy as np
from functools import partial
from gft.gft_internals import parse_eqn
from gft.gft_internals.gft_util import intern_labels, parse_model_specification, parse_metric_specification, better, checkpoint_filename, get_arg, set_arg

import logging
import math
import os
import random

import datasets,transformers
from datasets import load_dataset, load_metric

from tqdm.auto import tqdm

print(__name__ + ': loading from paddle', file=sys.stderr)

MAX_GPU_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32

from accelerate import Accelerator, DistributedType

from transformers import (
    AdamW,
    # AutoModelForSequenceClassification_hf,
    # AutoTokenizer_hf,
    get_linear_schedule_with_warmup,
    set_seed,
    AutoConfig,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler)

import paddle
from paddle.io import DataLoader as DataLoader_pd
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.transformers import (
    AutoModelForSequenceClassification as AutoModelForSequenceClassification_pd,
    AutoTokenizer as AutoTokenizer_pd,
)
from paddlenlp.transformers import LinearDecayWithWarmup

def save_all(model, tokenizer, fn):
    model.save_pretrained(fn)
    model.save_model_config(fn)
    tokenizer.save_pretrained(fn)

logger = logging.getLogger(__name__)

def fit(args, eqn, accelerator, raw_datasets):

    print('calling fit in fit_for_regress_pd.py (PaddleHub version)')
    assert eqn['eqn_type'] == parse_eqn.eqn_types['regress'], 'fit_for_regress is for eqn_type: regress, but eqn_type is: ' + str(
        parse_eqn.eqn_types_list[eqn['eqn_type']])
    
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

    dir = os.path.dirname(__file__)
    if dir == '': dir = ''

    x_field_names = eqn['x_field_names']
    y_field_names = eqn['y_field_names']

    interned_labels = label_list = None
    num_labels = len(y_field_names)

    model_provider,model_key = parse_model_specification(args)
    assert model_provider == 'PaddleHub', 'This case is for PaddleHub, not HuggingFace; provider = ' + str(model_provider)

    model = AutoModelForSequenceClassification_pd.from_pretrained(model_key, return_dict=True, num_labels=num_labels)
    tokenizer = AutoTokenizer_pd.from_pretrained(model_key)
    
    # def tokenize_function(examples):

    #     for y_field_name in y_field_names:
    #         assert y_field_name in examples, 'tokenize_function: cannot find %s in %s' % (y_field_name, str(examples))
    #     for x_field_name in x_field_names:
    #         assert x_field_name in examples, 'tokenize_function: cannot find %s in %s' % (x_field_name, str(examples))

    #     if len(x_field_names) == 1:
    #         outputs = tokenizer(examples[x_field_names[0]])
    #     else:
    #         outputs = tokenizer(examples[x_field_names[0]])

    #     outputs['labels'] = np.array([examples[y_field_name] for y_field_name in y_field_names],
    #                                  dtype=np.float32)

    #     return outputs


    def convert_example(example,
                        tokenizer,
                        label_list,
                        max_seq_length=512,
                        ):

        assert label_list is None, 'convert_example: confusion'

        label_dtype = 'float32'

        # assert len(y_field_names) == 1, 'convert_example: len(y_field_names) > 1, not yet supported'

        # raw_label = example[y_field_names[0]]
        # if isinstance(raw_label, list):
        #     label = [interned_labels[e] if e in interned_labels else e for e in raw_label]
        # else:
        #     label = interned_labels[raw_label] if raw_label in interned_labels else raw_label
        label = np.array([example[y_field_name] for y_field_name in y_field_names], dtype=label_dtype)

        if len(x_field_names) == 1:
            texts = (example[x_field_names[0]],)
        else:
            texts = (example[x_field_names[0]], example[x_field_names[1]])

        # example = tokenizer(*texts, max_seq_len=max_seq_length, pad_to_max_seq_len=get_arg(args, 'pad_to_max_length'))
        example = tokenizer(*texts)
        return example['input_ids'], example['token_type_ids'], label

    trans_func = partial(
        convert_example,
        tokenizer=tokenizer,
        label_list=label_list,
        max_seq_length=int(get_arg(args, 'max_length')))

    metric_provider,metric_key = parse_metric_specification(args)
    if metric_key is None:
            p = os.path.join(dir, 'sklearn_metrics/mean_squared_error.py')
            print('load_metric: ' + p)
            metric = load_metric(p)
            set_arg(args, 'better_figure_of_merit', -1) # less is more
            if not get_arg(args, 'figure_of_merit', default=None): 
                set_arg(args, 'figure_of_merit', 'mean_squared_error')
    else:
        print('load_metric: ' + metric_key)
        metric = load_metric(*metric_key.split(','))

    device = accelerator.device

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

    padding = "max_length" if get_arg(args, 'pad_to_max_length', default=False) else False

    def preprocess_function(examples):

        for y_field_name in y_field_names:
            assert y_field_name in examples, 'tokenize_function: cannot find %s in %s' % (y_field_name, str(examples))
        for x_field_name in x_field_names:
            assert x_field_name in examples, 'tokenize_function: cannot find %s in %s' % (x_field_name, str(examples))

        # Tokenize the texts
        if len(x_field_names) == 1: texts = (examples[x_field_names[0]],)
        else: texts = (examples[x_field_names[0]],examples[x_field_names[1]])

        # result = tokenizer(*texts, padding=padding, max_length=get_arg(args, 'max_length', default=128), truncation=True)
        # There doesn't seem to be a padding arg for the paddle tokenizer: padding=padding, 

        # pdb.set_trace()
        result = tokenizer(*texts)
        assert len(y_field_names) == 1, 'classify is currently limited to just one y variable: ' + str(y_field_names) + '; suggest using a HuggingFace model instead of PaddleHub'
        result['labels'] = examples[y_field_names[0]]
        # result['labels'] = np.array([examples[y_field_name] for y_field_name in y_field_names], dtype=np.float32)
        return result

    try:
        processed_datasets = raw_datasets.map(preprocess_function,
                                              batched=True,
                                              remove_columns=raw_datasets["train"].column_names,
                                              desc="Running tokenizer on dataset",)
    except:
        print('could not remove_columns=raw_datasets["train"].column_names')
        # processed_datasets = raw_datasets.map(preprocess_function,
        #                                       batched=True,
        #                                       desc="Running tokenizer on dataset",)
        assert 'val' in raw_datasets, 'cannot find validation set'
        processed_datasets = { 'train' : [preprocess_function(e) for e in raw_datasets['train']],
                               'val' : [preprocess_function(e) for e in raw_datasets['val']]}

    assert 'val' in processed_datasets, 'cannot find validation set'
    train_dataset = processed_datasets['train']
    eval_dataset = processed_datasets['val']

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    if get_arg(args, 'pad_to_max_length', default=False):
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None))

    bs = get_arg(args, 'per_device_train_batch_size', default=32)
    # train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=bs)
    # eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=bs)

    train_ds = raw_datasets['train']
    train_ds = train_ds.map(trans_func, lazy=True)
    dev_ds = raw_datasets['val']
    dev_ds = dev_ds.map(trans_func, lazy=True)

    train_batch_sampler = paddle.io.DistributedBatchSampler(
        train_ds, batch_size=batch_size, shuffle=True)
    dev_batch_sampler = paddle.io.BatchSampler(
        dev_ds, batch_size=batch_size, shuffle=False)

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # segment
        Stack(dtype="float32")  # label
    ): fn(samples)

    train_data_loader = DataLoader_pd(
        dataset=train_ds,
        batch_sampler=train_batch_sampler,
        collate_fn=batchify_fn,
        num_workers=0,
        return_list=True)

    dev_data_loader = DataLoader_pd(
        dataset=dev_ds,
        batch_sampler=dev_batch_sampler,
        collate_fn=batchify_fn,
        num_workers=0,
        return_list=True)

    max_train_samples = get_arg(args, 'max_train_samples', default=None)
    if max_train_samples is not None:
        num_training_steps = int(max_train_samples)
        num_train_epochs = math.ceil(num_training_steps / len(train_data_loader))
    else:
        num_training_steps = len(train_data_loader) * num_epochs
        num_train_epochs = num_epochs

    warmup = int(get_arg(args, 'num_warmup_steps', default=0))

    lr_scheduler = LinearDecayWithWarmup(lr, num_training_steps, warmup)

    # Generate parameter names needed to perform weight decay.
    # All bias and LayerNorm parameters are excluded.
    decay_params = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        parameters=model.parameters(),
        weight_decay=float(get_arg(args, 'weight_decay', default=0.0)),
        apply_decay_param_fun=lambda x: x in decay_params)

    # loss_fct = paddle.nn.loss.CrossEntropyLoss(
    # ) if label_list else paddle.nn.loss.MSELoss()

    loss_fct = paddle.nn.loss.MSELoss()

    global_step = 0
    best_so_far = None
    for epoch in range(num_train_epochs):
        for step, batch in enumerate(train_data_loader):
            global_step += 1
            if global_step >= num_training_steps:
                return
            input_ids, segment_ids, labels = batch
            logits = model(input_ids, segment_ids)
            loss = loss_fct(logits, labels)

            # print('labels: ' + str(labels))
            # print('logits: ' + str(logits))

            # import pdb
            # pdb.set_trace()
            # loss = torch.nn.functional.mse_loss(logits.logits, batch['labels']).float()
            # loss = loss_fct(logits.view(-1), labels.view(-1)).float()
            # loss.float().backward()

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.clear_grad()

        model.eval()
        time0 = time.time()
        with paddle.no_grad():
            for batch in dev_data_loader:
                input_ids, segment_ids, labels = batch
                logits = model(input_ids, segment_ids)
                predictions = logits.argmax(axis=-1).numpy()
                gold_references = labels.squeeze().numpy()
                metric.add_batch(predictions=predictions, references=gold_references)
        eval_metric = metric.compute()
        logger.info(f"epoch {epoch}: {eval_metric} {time.time() - time0} seconds")
        model.train()

        fn = checkpoint_filename(args, model_key, epoch, False)
        b, best = better(args, eval_metric, best_so_far)
        if b:
            best_so_far = best
            fig = str(get_arg(args, 'figure_of_merit', default=None))
            bfig = str(get_arg(args, 'better_figure_of_merit', default=None))
            print('best_so_far: %f (figure_of_merit = %s, better_figure_of_merit = %s)' % (best_so_far, fig, bfig))
            fn = checkpoint_filename(args, model_key, epoch, True)
        if not get_arg(args, 'output_dir') is None:
            prev = checkpoint_filename(args, model_key, epoch-1, False)
            try:
                if os.path.exists(prev): shutil.rmtree(prev)
            except:
                print('failed to delete: ' + str(prev), sys.stderr)

            # save_all(model, tokenizer, fn)
            if paddle.distributed.get_rank() == 0:
                model_to_save = model._layers if isinstance(
                    model, paddle.DataParallel) else model
                save_all(model_to_save, tokenizer, fn)

    # # Optimizer
    # # Split weights in two groups, one with weight decay and the other not.
    # no_decay = ["bias", "LayerNorm.weight"]
    # optimizer_grouped_parameters = [
    #     {
    #         "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
    #         "weight_decay": get_arg(args, 'weight_decay', default=0.0),
    #     },
    #     {
    #         "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
    #         "weight_decay": 0.0,
    #     },
    # ]
    # optimizer = AdamW(optimizer_grouped_parameters, lr=lr)

    # # Prepare everything with our `accelerator`.
    # model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(model, optimizer, train_dataloader, eval_dataloader)

    # # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
    # # shorter in multiprocess)

    # # Scheduler and math around the number of training steps.
    # gas = get_arg(args, 'gradient_accumulation_steps', default=1)
    # num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gas)
    # epochs = int(get_arg(args, 'num_train_epochs', default=15))
    # max_train_steps = get_arg(args, 'max_train_steps', default=None)
    # if max_train_steps is None:
    #     max_train_steps = epochs * num_update_steps_per_epoch
    # else:
    #     epochs = int(math.ceil(max_train_steps / num_update_steps_per_epoch))

    # lr_scheduler = get_scheduler(
    #     name=get_arg(args, 'lr_scheduler_type', default="linear"),
    #     optimizer=optimizer,
    #     num_warmup_steps=get_arg(args, 'num_warmup_steps', default=0),
    #     num_training_steps=max_train_steps,
    # )

    # # Train!

    # total_batch_size = bs * accelerator.num_processes * gas

    # logger.info("***** Running training *****")
    # logger.info(f"  Num examples = {len(train_dataset)}")
    # logger.info(f"  Num Epochs = {epochs}")
    # logger.info(f"  Instantaneous batch size per device = {bs}")
    # logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    # logger.info(f"  Gradient Accumulation steps = {gas}")
    # logger.info(f"  Total optimization steps = {max_train_steps}")
    # completed_steps = 0

    # best_so_far = None
    # time0 = time.time()

    # for epoch in range(int(epochs)):
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
    #             completed_steps += 1

    #         if completed_steps >= max_train_steps:
    #             break

    #     model.eval()
    #     for step, batch in enumerate(eval_dataloader):
    #         outputs = model(**batch)
    #         predictions = outputs.logits.argmax(dim=-1) if not is_regression else outputs.logits.squeeze()
    #         metric.add_batch(predictions=accelerator.gather(predictions),
    #                          references=accelerator.gather(batch["labels"]))

    #     eval_metric = metric.compute()
    #     logger.info(f"epoch {epoch}: {eval_metric} {time.time() - time0} seconds")
    #     accelerator.print('%0.0f seconds: epoch %d validation %s' % (time.time() - time0, epoch, str(eval_metric)))

    #     fn = checkpoint_filename(args, model_key, epoch, False)
    #     b,best = better(args, eval_metric, best_so_far)
    #     if b:
    #         best_so_far = best
    #         fig = str(get_arg(args, 'figure_of_merit', default=None))
    #         bfig = str(get_arg(args, 'better_figure_of_merit', default=None))
    #         print('best_so_far: %f (figure_of_merit = %s, better_figure_of_merit = %s)' % (best_so_far, fig, bfig))
    #         fn = checkpoint_filename(args, model_key, epoch, True)
    #     if not get_arg(args, 'output_dir') is None:
    #         prev = checkpoint_filename(args, model_key, epoch-1, False)
    #         try:
    #             if os.path.exists(prev): shutil.rmtree(prev)
    #         except:
    #             print('failed to delete: ' + str(prev), sys.stderr)
    #         model.save_pretrained(fn)

    # # if args.checkpoint is not None:
    # #     accelerator.wait_for_everyone()
    # #     unwrapped_model = accelerator.unwrap_model(model)
    # #     unwrapped_model.save_pretrained(args.checkpoint, save_function=accelerator.save)

