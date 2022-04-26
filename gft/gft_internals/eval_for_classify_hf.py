
# based on https://github.com/huggingface/transformers/blob/master/examples/pytorch/text-classification/run_glue_no_trainer.py

# convention: _hf is from HuggingFace, and _pd is from PaddleHub

# import pdb

import gft
import sys,os,shutil,torch,time
from torch.utils.data import DataLoader
import numpy as np
from gft.gft_internals import parse_eqn
from gft.gft_internals.gft_util import intern_labels, parse_model_specification, parse_base_model_specification, parse_metric_specification, parse_dataset_specification, better, checkpoint_filename, get_arg, set_arg

import logging
import math
import os
import random

import datasets,transformers
from datasets import load_dataset, load_metric

# from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm


MAX_GPU_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 16

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

from transformers import AutoModelForSequenceClassification as AutoModelForSequenceClassification_hf
from transformers import AutoTokenizer as AutoTokenizer_hf


logger = logging.getLogger(__name__)

def my_eval(args, eqn, accelerator, raw_datasets, is_regression=False):

    print('calling my_eval in eval_for_classify_hf.py (HuggingFace version)', file=sys.stderr)
    if is_regression:
        assert eqn['eqn_type'] == parse_eqn.eqn_types['regress'], 'eval_for_classify is for eqn_type: regress, but eqn_type is: ' + str(
            parse_eqn.eqn_types_list[eqn['eqn_type']])
    else:
        assert eqn['eqn_type'] == parse_eqn.eqn_types['classify'], 'eval_for_classify is for eqn_type: classify, but eqn_type is: ' + str(
            parse_eqn.eqn_types_list[eqn['eqn_type']])

    # If passed along, set the training seed now.
    seed = get_arg(args, 'seed', default=None)
    print('seed: ' + str(seed), file=sys.stderr)
    if not seed is None:
        set_seed(seed)

    # Sample hyper-parameters for learning rate, batch size, seed and a few other HPs
    lr = get_arg(args, 'learning_rate', default=5e-5)
    num_epochs = int(get_arg(args, 'num_train_epochs', default=15))
    # correct_bias = config["correct_bias"]
    batch_size = int(get_arg(args, 'per_device_train_batch_size', default=MAX_GPU_BATCH_SIZE))

    dir = os.path.dirname(__file__)
    if dir == '': dir = '.'

    x_field_names = eqn['x_field_names']
    y_field_names = eqn['y_field_names']

    # import pdb
    # pdb.set_trace()

    base_model_provider, base_model_key, model_provider, model_key = parse_base_model_specification(args)

    # model_provider,model_key = parse_model_specification(args, keyword='model')
    # base_model_provider,base_model_key = parse_model_specification(args, keyword='base_model')
    assert model_provider == 'HuggingFace' or model_provider == 'Custom', 'Expected provider to be HuggingFace or Custom, but it was: ' + str(model_provider)

    data_provider,data_key = parse_dataset_specification(args)
    assert not (data_provider == 'PaddleHub' and is_regression), 'case not yet supported: regression on model_provider == HuggingFace and data_provider == PaddleHub'

    print('model_key: ' + str(model_key), file=sys.stderr)

    from gft.gft_internals.my_auto_model_hf import my_load_model_tokenizer_and_extractor
    model,tokenizer,extractor = my_load_model_tokenizer_and_extractor(args)

    print('tokenizer: ' + str(tokenizer), file=sys.stderr)

    from gft.gft_internals.gft_util import labels_from_model
    labs = labels_from_model(model)

    if not is_regression:
        interned_labels,label_list = intern_labels(raw_datasets, y_field_names, args, labs=labs)
        num_labels = len(interned_labels)
    else:
        interned_labels = label_list = None
        num_labels = len(y_field_names)

    # model = AutoModelForSequenceClassification_hf.from_pretrained(model_key, return_dict=True, num_labels=num_labels)
    # if not base_model_key is None: 
    #     tokenizer = AutoTokenizer_hf.from_pretrained(base_model_key)
    # else: 
    #     tokenizer = AutoTokenizer_hf.from_pretrained(model_key)

    metric_provider,metric_key = parse_metric_specification(args)

    if metric_key is None:
        if is_regression:
            p = os.path.join(dir, 'sklearn_metrics/mean_squared_error.py')
            print('load_metric: ' + p, file=sys.stderr)
            metric = load_metric(p)
            set_arg(args, 'better_figure_of_merit', -1) # less is more
            if not get_arg(args, 'figure_of_merit', default=None): 
                set_arg(args, 'figure_of_merit', 'mean_squared_error')
        else:
            p = os.path.join(dir, 'sklearn_metrics/multiclass_glue.py')
            print('load_metric: ' + p + ',mrpc', file=sys.stderr)
            metric = load_metric(p,  'mrpc')
    else:
        print('load_metric: ' + metric_key, file=sys.stderr)
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

        assert not tokenizer is None, 'expected tokenizer to be something other than None'

        result = tokenizer(*texts, padding=padding, max_length=get_arg(args, 'max_length', default=128), truncation=True)

        assert len(y_field_names) == 1, 'classify is currently limited to just one y variable: ' + str(y_field_names)
        if interned_labels is None:
            result['labels'] = [e for e in examples[y_field_names[0]]]
        else:
            ys = examples[y_field_names[0]]
            if isinstance(ys, list):
                result['labels'] = [interned_labels[e] if e in interned_labels else e
                                    for e in ys]
            else:
                e = ys
                result['labels'] = interned_labels[e] if e in interned_labels else e
        return result

    val_split = get_arg(args, 'split', default='val')

    try:
        processed_datasets = raw_datasets.map(preprocess_function,
                                              batched=True,
                                              remove_columns=raw_datasets["train"].column_names,
                                              desc="Running tokenizer on dataset",)
    except:
        print('warning could not remove_columns=raw_datasets["train"].column_names', file=sys.stderr)
        # processed_datasets = raw_datasets.map(preprocess_function,
        #                                       batched=True,
        #                                       desc="Running tokenizer on dataset",)
        assert val_split in raw_datasets, 'cannot find validation set, val_split: ' + str(val_split)
        processed_datasets = { 'train' : [preprocess_function(e) for e in raw_datasets['train']],
                               val_split : [preprocess_function(e) for e in raw_datasets[val_split]]}



    # assert 'val' in processed_datasets, 'cannot find validation set'
    train_dataset = processed_datasets['train']
    eval_dataset = processed_datasets[val_split]

    # Log a few random samples from the training set:
    # for index in random.sample(range(len(train_dataset)), 3):
    #     logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

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

    bs = get_arg(args, 'per_device_train_batch_size', default=MAX_GPU_BATCH_SIZE)
    ebs = get_arg(args, 'per_device_eval_batch_size', default=EVAL_BATCH_SIZE)
    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=bs)
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

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(model, optimizer, train_dataloader, eval_dataloader)

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

    lr_scheduler = get_scheduler(
        name=get_arg(args, 'lr_scheduler_type', default="linear"),
        optimizer=optimizer,
        num_warmup_steps=get_arg(args, 'num_warmup_steps', default=0),
        num_training_steps=max_train_steps,
    )

    # Train!

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
    time0 = time.time()

    model.eval()
    for step, batch in enumerate(eval_dataloader):
        outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1) if not is_regression else outputs.logits.squeeze()
        metric.add_batch(predictions=accelerator.gather(predictions),
                         references=accelerator.gather(batch["labels"]))
        # except:
        #     print('error in batch: ' + str(batch), file=sys.stderr)
        
    eval_metric = metric.compute()
    logger.info(f"{eval_metric} {time.time() - time0} seconds")
    fig = get_arg(args, 'figure_of_merit', default=None)

    if fig is None:
        res = '\t'.join(['%s: %s' % (k, str(eval_metric[k])) for k in eval_metric])
    else:
        res = '%s: %s' % (fig, str(eval_metric[fig]))

    accelerator.print('\t'.join(map(str, ['%0.2f seconds' % (time.time() - time0), model_key, res])))
    
