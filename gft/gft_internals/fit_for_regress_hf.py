import gft
import sys,os,shutil,argparse,time,torch
from torch.utils.data import DataLoader
import numpy as np
from gft.gft_internals import parse_eqn
from gft.gft_internals.gft_util import intern_labels, parse_model_specification, parse_metric_specification, better, checkpoint_filename, get_arg, set_arg

# like fit_for_classify.py, but generalizes to lhs with multiple variables
# example: regress: Valence + Arousal + Dominance ~ Word

from accelerate import Accelerator, DistributedType

from datasets import load_dataset, load_metric
from transformers import (
    AdamW,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    set_seed,
)

MAX_GPU_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32

def fit(args, eqn, accelerator, datasets):

    print('calling fit in fit_for_regression.py')
    assert eqn['eqn_type'] == parse_eqn.eqn_types['regress'], 'fit_for_regress is for eqn_type: regress, but eqn_type is: ' + str(eqn['eqn_type'])

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

    provider,model_key = parse_model_specification(args)
    assert not provider == 'PaddlePaddle', 'PaddlePaddle models are not supported (yet)' # stub to flesh out later

    model = AutoModelForSequenceClassification.from_pretrained(model_key, return_dict=True, num_labels=len(y_field_names))
    tokenizer = AutoTokenizer.from_pretrained(model_key)

    metric_provider,metric_key = parse_metric_specification(args)
    assert not metric_provider == 'PaddlePaddle', 'PaddlePaddle metrics are not supported (yet)' # stub to flesh out later

    if metric_key is None:
        p = os.path.join(dir, 'sklearn_metrics/mean_squared_error.py')
        print('load_metric: ' + p)
        val_metric = load_metric(p)
        train_metric = load_metric(p)
        set_arg(args, 'better_figure_of_merit', -1) # less is more
        if not get_arg(args, 'figure_of_merit', default=None): 
            set_arg(args, 'figure_of_merit', 'mean_squared_error')
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
            outputs = tokenizer(examples[x_field_names[0]])
        else:
            outputs = tokenizer(examples[x_field_names[0]], examples[x_field_names[1]])

        # if len(x_field_names) == 1:
        #     outputs = tokenizer(examples[x_field_names[0]], truncation=True, max_length=None)
        # else:
        #     outputs = tokenizer(examples[x_field_names[0]], examples[x_field_names[1]], truncation=True, max_length=None)

        outputs['labels'] = np.array([examples[y_field_name] for y_field_name in y_field_names],
                                     dtype=np.float32)

        return outputs

    assert 'val' in datasets, 'cannot find validation set'
    tokenized_datasets = { 'train' : [tokenize_function(e) for e in datasets['train']],
                           'validation' : [tokenize_function(e) for e in datasets['val']],
                           # 'test' : [tokenize_function(e) for e in datasets['test']]
    }

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
        tokenized_datasets["train"], shuffle=True, collate_fn=collate_fn, batch_size=batch_size
    )
    eval_dataloader = DataLoader(
        tokenized_datasets["validation"], shuffle=False, collate_fn=collate_fn, batch_size=EVAL_BATCH_SIZE
    )

    # Instantiate the model (we build the model here so that the seed also control new weights initialization)
    # model = AutoModelForSequenceClassification.from_pretrained(args.pretrained, return_dict=True)
    # model = AutoModelForMultipleChoice.from_pretrained(args.pretrained, return_dict=True)

    # We could avoid this line since the accelerator is set with `device_placement=True` (default value).
    # Note that if you are placing tensors on devices manually, this line absolutely needs to be before the optimizer
    # creation otherwise training will not work on TPU (`accelerate` will kindly throw an error to make us aware of that).
    model = model.to(accelerator.device)

    # Instantiate optimizer
    optimizer = AdamW(params=model.parameters(), lr=lr)

    # Prepare everything
    # There is no specific order to remember, we just need to unpack the objects in the same order we gave them to the
    # prepare method.
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    # Instantiate learning rate scheduler after preparing the training dataloader as the prepare method
    # may change its length.
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=len(train_dataloader) * num_epochs,
    )

    # def checkpoint_filename(epoch, best):
    #     if args.checkpoint is None: return None
    #     suffix = 'best'
    #     if not best is True:
    #         suffix = 'epoch.' + str(epoch)
    #     return '%s/%s/fit_multiclass.%s' % (args.checkpoint, model_key, suffix)

    # def better(metric, best_so_far):

    #     fig = args.figure_of_merit
    #     better_direction = args.better_figure_of_merit

    #     if fig is None:
    #         fig = 'mean_squared_error'
    #         better_direction = -1

    #     if best_so_far is None: 
    #         return True, metric[fig]

    #     if better_direction > 0:
    #         return (metric[fig] > best_so_far), metric[fig]
    #     else: 
    #         return (metric[fig] < best_so_far), metric[fig]

    def train_loop_with_regression(epoch):
        eval_steps = get_arg(args, 'eval_steps', default=None)
        if eval_steps is None:
            eval_steps = 1000
        eval_steps = int(eval_steps)

        model.train()
        for step, batch in enumerate(train_dataloader):
            # We could avoid this line since we set the accelerator with `device_placement=True`.
            batch.to(accelerator.device)
            outputs = model(**batch)
            loss = torch.nn.functional.mse_loss(outputs.logits.view(-1), batch['labels'].view(-1)).float()
            loss = loss/gradient_accumulation_steps
            accelerator.backward(loss.float())
            if step % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            if step % eval_steps == 0:
                accelerator.print("step %d of %d"% (step, len(train_dataloader)))
                sys.stdout.flush()

    def eval_loop_with_regression(epoch, best_so_far):
        model.eval()
        for step, batch in enumerate(eval_dataloader):
            # We could avoid this line since we set the accelerator with `device_placement=True`.
            batch.to(accelerator.device)
            with torch.no_grad():
                outputs = model(**batch)
            predictions = outputs.logits
            val_metric.add_batch(predictions=accelerator.gather(predictions.view(-1)), references=accelerator.gather(batch["labels"].view(-1)))

        eval_metric = val_metric.compute()
        # Use accelerator.print to print only on the main process.
        accelerator.print(f"epoch {epoch} validation:", eval_metric)

        for step, batch in enumerate(train_dataloader):
            # We could avoid this line since we set the accelerator with `device_placement=True`.
            batch.to(accelerator.device)
            with torch.no_grad():
                outputs = model(**batch)
            
        predictions = outputs.logits
        train_metric.add_batch(predictions=accelerator.gather(predictions.view(-1)), references=accelerator.gather(batch["labels"].view(-1)))

        eval_metric = train_metric.compute()
        # Use accelerator.print to print only on the main process.
        accelerator.print(f"epoch {epoch} train:", eval_metric)

        fn = checkpoint_filename(args, model_key, epoch, False)
        b,best = better(args, eval_metric, best_so_far)
        if b is True:
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
    fig = str(get_arg(args, 'figure_of_merit', default=None))
    bfig = str(get_arg(args, 'better_figure_of_merit', default=None))
    for epoch in range(num_epochs):        
        print('about to train epoch %d: %0.0f sec' % (epoch, time.time() - time0))
        train_loop_with_regression(epoch)
        best_so_far = eval_loop_with_regression(epoch, best_so_far)
        print('about to eval epoch %d: %0.0f sec' % (epoch, time.time() - time0))
        print('epoch: %d, best_so_far: %s (figure_of_merit = %s, better_figure_of_merit = %s)' %  (epoch, str(best_so_far), fig, bfig))


