import gft
import sys,os,shutil,argparse,torch,time
from torch.utils.data import DataLoader
import numpy as np
from gft.gft_internals import parse_eqn
from gft.gft_internals.gft_util import intern_labels, parse_model_specification, parse_metric_specification, better, checkpoint_filename, get_arg, set_arg

MAX_GPU_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32

from accelerate import Accelerator, DistributedType

from datasets import load_dataset, load_metric
from transformers import (
    AdamW,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    set_seed,
)

def fit(args, eqn, accelerator, datasets):

    print('calling fit in fit_for_classify.py')
    assert eqn['eqn_type'] == parse_eqn.eqn_types['classify_classic'], 'fit_for_classify is for eqn_type: classify, but eqn_type is: ' + str(
        parse_eqn.eqn_types_list[eqn['eqn_type']])

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
    interned_labels,label_list = intern_labels(datasets, y_field_names, args)

    provider,model_key = parse_model_specification(args)
    assert not provider == 'PaddlePaddle', 'PaddlePaddle models are not supported (yet)' # stub to flesh out later

    model = AutoModelForSequenceClassification.from_pretrained(model_key, return_dict=True, num_labels=len(interned_labels))
    tokenizer = AutoTokenizer.from_pretrained(model_key)
    
    metric_provider,metric_key = parse_metric_specification(args)
    assert not metric_provider == 'PaddlePaddle', 'PaddlePaddle metrics are not supported (yet)' # stub to flesh out later

    if metric_key is None:
        p = os.path.join(dir, 'sklearn_metrics/multiclass_glue.py')
        print('load_metric: ' + p + ',mrpc')
        val_metric = load_metric(p, 'mrpc')
        train_metric = load_metric(p, 'mrpc')
    else:
        print('load_metric: ' + metric_key)
        val_metric = load_metric(*metric_key.split(','))
        train_metric = load_metric(*metric_key.split(','))

    def tokenize_function(examples):

        for y_field_name in y_field_names:
            assert y_field_name in examples, 'tokenize_function: cannot find %s in %s' % (y_field_name, str(examples))
        for x_field_name in x_field_names:
            assert x_field_name in examples, 'tokenize_function: cannot find %s in %s' % (x_field_name, str(examples))

        if len(x_field_names) == 1:
            outputs = tokenizer(examples[x_field_names[0]], truncation=True, max_length=None)
        else:
            outputs = tokenizer(examples[x_field_names[0]], examples[x_field_names[1]], truncation=True, max_length=None)

        assert len(y_field_names) == 1, 'classify is currently limited to just one y variable: ' + str(y_field_names)
        outputs['labels'] = interned_labels[examples[y_field_names[0]]]
        return outputs

    assert 'val' in datasets, 'cannot find validation set'
    tokenized_datasets = { 'train' : [tokenize_function(e) for e in datasets['train']],
                           'validation' : [tokenize_function(e) for e in datasets['val']]}

    gradient_accumulation_steps = 1
    if batch_size > MAX_GPU_BATCH_SIZE:
        gradient_accumulation_steps = batch_size // MAX_GPU_BATCH_SIZE
        batch_size = MAX_GPU_BATCH_SIZE

    def collate_fn(examples):
        # On TPU it's best to pad everything to the same length or training will be very slow.
        if accelerator.distributed_type == DistributedType.TPU:
            # return tokenizer.pad(examples, padding="max_length", max_length=128, return_tensors="pt")
            return tokenizer.pad(examples, padding="max_length", max_length=12, return_tensors="pt") # changed by kwc
        return tokenizer.pad(examples, padding="longest", return_tensors="pt")

    # Instantiate dataloaders.
    train_dataloader = DataLoader(
        tokenized_datasets["train"], shuffle=True, collate_fn=collate_fn, batch_size=batch_size)
    eval_dataloader = DataLoader(
        tokenized_datasets["validation"], shuffle=False, collate_fn=collate_fn, batch_size=EVAL_BATCH_SIZE)

    # We could avoid this line since the accelerator is set with `device_placement=True` (default value).
    # Note that if you are placing tensors on devices manually, this line absolutely needs to be before the optimizer
    # creation otherwise training will not work on TPU (`accelerate` will kindly throw an error to make us aware of that).
    model = model.to(accelerator.device)

    # Instantiate optimizer
    # optimizer = AdamW(params=model.parameters(), lr=lr, correct_bias=correct_bias)
    optimizer = AdamW(params=model.parameters(), lr=lr)

    # Prepare everything
    # There is no specific order to remember, we just need to unpack the objects in the same order we gave them to the
    # prepare method.
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader)

    # Instantiate learning rate scheduler after preparing the training dataloader as the prepare method
    # may change its length.
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=len(train_dataloader) * num_epochs)

    def train_loop_with_classification(epoch):
        print('train_loop, epoch: ' + str(epoch))

        eval_steps = get_arg(args, 'eval_steps', default=None)
        if eval_steps is None:
            eval_steps = 1000
        eval_steps = int(eval_steps)

        model.train()
        for step, batch in enumerate(train_dataloader):
            # We could avoid this line since we set the accelerator with `device_placement=True`.
            batch.to(accelerator.device)
            # pdb.set_trace()
            outputs = model(**batch)
            loss = outputs.loss
            loss = loss / gradient_accumulation_steps
            accelerator.backward(loss)
            if step % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            if step % eval_steps == 0:
                accelerator.print("step %d of %d"% (step, len(train_dataloader)))
                sys.stdout.flush()

    def eval_loop_with_classification(epoch, best_so_far):

        print('eval_loop, epoch: ' + str(epoch))

        model.eval()
        for step, batch in enumerate(eval_dataloader):
            # We could avoid this line since we set the accelerator with `device_placement=True`.
            batch.to(accelerator.device)
            with torch.no_grad():
                outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            val_metric.add_batch(predictions=accelerator.gather(predictions), references=accelerator.gather(batch['labels']))

        eval_metric = val_metric.compute()
        # Use accelerator.print to print only on the main process.
        accelerator.print(f"epoch {epoch} validation:", eval_metric)

        for step, batch in enumerate(train_dataloader):
            # We could avoid this line since we set the accelerator with `device_placement=True`.
            batch.to(accelerator.device)
            with torch.no_grad():
                outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            train_metric.add_batch(predictions=accelerator.gather(predictions), references=accelerator.gather(batch['labels']))

        eval_metric = train_metric.compute()
        # Use accelerator.print to print only on the main process.
        accelerator.print(f"epoch {epoch} train:", eval_metric)

        fn = checkpoint_filename(args, model_key, epoch, False)
        b,best = better(args, eval_metric, best_so_far)
        if b:
            best_so_far = best
            fn = checkpoint_filename(args, model_key, epoch, True)
        if not get_arg(args, 'output_dir') is None:
            prev = checkpoint_filename(args, model_key, epoch-2, False)
            try:
                if os.path.exists(prev): shutil.rmtree(prev)
            except:
                print('failed to delete: ' + str(prev), sys.stderr)
            model.save_pretrained(fn)
        return best_so_far

    best_so_far = None
    time0 = time.time()
    for epoch in range(num_epochs):        
        print('about to train epoch %d: %0.0f sec' % (epoch, time.time() - time0))
        sys.stdout.flush()        

        train_loop_with_classification(epoch)

        print('about to eval epoch %d: %0.0f sec' % (epoch, time.time() - time0))
        sys.stdout.flush()
        best_so_far = eval_loop_with_classification(epoch, best_so_far)
