# convention: _hf is from HuggingFace, and _pd is from PaddleHub
import gft
import numpy as np
import os,sys,json,time

print(__name__ + ': loading from huggingface', file=sys.stderr)

t0 = time.time()

from gft.gft_internals.gft_util import parse_model_specification,parse_dataset_specification,parse_task_specification,get_arg
from gft.gft_internals import my_datasets
from gft.gft_internals import my_auto_model_hf

def f2s(f):
    return str(f)

def floats2str(fs):
    return '|'.join(map(str, fs))

def collate_fn(tokenizer, examples):
    return tokenizer.pad(examples, padding="longest", return_tensors="pt")

def get_config(fn):
    with open(fn + '/config.json', 'r') as fd:
        return json.loads(fd.read())

labels = '***undefined***'

def best_label(logits):
    res = np.argmax(logits)
    if labels is None: return str(res)
    else: return str(res) + '/' + str(labels[res])

def labels_from_args(args):

    dataset_provider,dataset_key = parse_dataset_specification(args)
    if not dataset_key is None:
        ds = my_datasets.my_load_dataset(args)
        for split in ds:
            if hasattr(ds[split], 'features'):
                for col in ds[split].features:
                    if hasattr(ds[split].features[col], 'names'):
                        return ds[split].features[col].names

    labels = get_arg(args, 'labels', default=None)

    if labels is None:
        model_provider,model_key = parse_model_specification(args, keyword='model')
        if not model_key is None:
            p = os.path.join(model_key + 'labels')
            if os.path.exists(p): labels = p

    if not labels is None:
        with open(labels, 'r') as fd:
            labels = fd.read().split('\n')

    return labels

def apply_pipeline(args, xfields, pipe, task, model, tokenizer):
    
    if get_arg(args, 'do_not_catch_errors', default=False):
        return apply_pipeline_internal(args, xfields, pipe, task, model, tokenizer), 0
    else:
        try: return apply_pipeline_internal(args, xfields, pipe, task, model, tokenizer), 0
        except: return '***ERROR***', 1

def apply_pipeline_internal(args, xfields, pipe, task, model, tokenizer):
    # print('xfields: ' + str(xfields), file=sys.stderr)
    # pdb.set_trace()

    if task is None:
        parsed = collate_fn(tokenizer, [tokenizer(*xfields, truncation=True, max_length=None)])
        outputs = model(**parsed)
        logits = outputs['logits'].detach().numpy()[0]
        return best_label(logits) + '\t' + floats2str(logits)

    if task == "audio-classification":
        assert False, 'apply_pipeline_internal, task not supported: ' + task

    if task == "ASR" or task == "automatic-speech-recognition":
        res = pipe(xfields[0])
        return res['text']

    if task == "conversational":
        assert False, 'apply_pipeline_internal, task not supported: ' + task

    if task == "feature-extraction":
        assert False, 'apply_pipeline_internal, task not supported: ' + task

    if task == "fill-mask":

        # The mask_token depends on the tokenizer,
        # but we would like to hide that complexity 
        # from the users (as much as possible)

        if hasattr(pipe, 'tokenizer') and hasattr(pipe.tokenizer, 'mask_token'):
            mask_token = pipe.tokenizer.mask_token
        else: mask_token = "<mask>"

        if mask_token == "<mask>":
            res = pipe(xfields[0])
        else:
            res = pipe(xfields[0].replace("<mask>", mask_token))
        return '\t'.join(['%s|%0.3f' % (r['token_str'], r['score']) for r in res ])

    if task == "image-classification":
        res = pipe(xfields[0])
        return '\t'.join(['%s|%0.3f' % (r['label'], r['score']) for r in res ])

    if task == "QA" or task == "question-answering":
        assert len(xfields) == 2, 'expected 2 fields: found ' + str(len(xfields))
        question,context=xfields[0:2]
        res = pipe({'question' : question, 'context' : context})
        return 'answer: ' + res['answer'] + '\tscore: %0.4f' % res['score'] + ' span: %d-%d' % (res['start'], res['end'])

    if task == "table-question-answering":
        assert False, 'apply_pipeline_internal, task not supported: ' + task

    if task == "text2text-generation":
        res = pipe(*xfields)
        return '\t'.join([r['generated_text'] for r in res])
        # assert False, 'apply_pipeline_internal, task not supported: ' + task

    if task == "text-classification" or task == "sentiment-analysis":
        texts = [xfields[0]]
        # print('texts: ' + str(texts), file=sys.stderr)
        res = pipe(texts, max_length=512)
        # print('res: ' + str(res), file=sys.stderr)
        return str(res[0]['label']) + '\t' + str(res[0]['score'])

    if task == "text-generation":
        res = pipe(xfields[0], max_length=get_arg(args, 'max_length', default=30), num_return_sequences=get_arg(args, 'num_return_sequences', default=3))
        return '\t'.join([r['generated_text'] for r in res ])

    if task == "token-classification" or task == "ner":
        x = xfields[0]
        if isinstance(x, list):
            x = ' '.join(x)
        res = pipe(x)
        return '\t'.join(['%s/%s:%0.4f' % (r['word'], r['entity'], r['score']) for r in res])

    if task == "MT" or task == "translation":
        res = pipe(*xfields)
        return res[0]['translation_text']

    if task == "translation_xx_to_yy":
        assert False, 'apply_pipeline_internal, task not supported: ' + task

    if task == "summarization":
        assert False, 'apply_pipeline_internal, task not supported: ' + task

    if task == "zero-shot-classification":
        assert False, 'apply_pipeline_internal, task not supported: ' + task

    assert False, 'apply_pipeline_internal, task not supported: ' + task

def ds_from_args(args):
    from gft.gft_internals.my_datasets import my_load_dataset
    if get_arg(args, 'data', default=None) is None:
        print('ds_from_args: cannot find data', file=sys.stderr)
        return None
    ds = my_load_dataset(args)
    split = get_arg(args, 'split', default=None)
    if split in ds: return ds[split]
    else: 
        print('ds_from_args: cannot find split (%s) in data' % str(split), file=sys.stderr)
        return None    

def gft_predict_hf_with_pipeline(args):
    from transformers import pipeline
    model_provider,model_key = parse_model_specification(args, keyword='model')
    assert model_provider != 'PaddleNLP', 'expected provider to be HuggingFace; provider: ' + str(model_provider)

    delim = get_arg(args, 'delimiter')
    task_provider,task = parse_task_specification(args)
    from gft.gft_internals.my_task import infer_task
    task = infer_task(args)

    # if task is None:
    #     from parse_eqn import infer_task_from_eqn
    #     task = infer_task_from_eqn(args)

    pipe = None
    model = None
    tokenizer = None
    extractor = None

    try:
        # This calls auto_model_for_X, which can signal errors
        # because we do not have auto model classes for all cases
        model,tokenizer,extractor = my_auto_model_hf.my_load_model_tokenizer_and_extractor(args, 'model')
        
        if not model is None and tokenizer is None:
            from transformers import AutoFeatureExtractor
            extractor = AutoFeatureExtractor.from_pretrained(model_key)

    except:
        tokenizer = model = model_key

    if task is None:
        pipe = None
    elif model is None:
        print('task: ' + str(task), file=sys.stderr)
        pipe = pipeline(task)
    elif tokenizer is None:
        assert not extractor is None, 'confusion in gft_predict_hf_with_pipeline'
        pipe = pipeline(task, model=model, feature_extractor=extractor)
    else:
        pipe = pipeline(task, model=model, tokenizer=tokenizer)
        
    ds = ds_from_args(args)
    eqn = get_arg(args, 'eqn')
    line_number=0
    errors=0

    if ds is None:
        print('about to read from sys.stdin', file=sys.stderr)
        for line in sys.stdin:
            line_number +=1
            fields = line.rstrip().split(delim)
            xfields = None

            if len(fields) == 0: continue

            if len(fields) >= 1: 
                xfields = fields[0].split('|')

            res,err = apply_pipeline(args, xfields, pipe, task, model, tokenizer)
            errors += err
            print(line.rstrip() + delim + res)
            sys.stdout.flush()

    else:
        from gft.gft_internals.parse_eqn import parse_eqn
        eqn = parse_eqn(eqn)
        assert not eqn is None, 'gft_predict: if --data argument is specified, then --eqn argument is required'
        for record in ds:
            line_number +=1
            yfields = [record[f] for f in eqn['y_field_names']]
            xfields = [record[f] for f in eqn['x_field_names']]
            res,err = apply_pipeline(args, xfields, pipe, task, model, tokenizer)
            errors += err
            print(delim.join(['|'.join(map(str, xfields)), 
                              '|'.join(map(str, yfields)), 
                              res]))
            sys.stdout.flush()
    print('total time: %0.2f seconds; processed %d input lines; caught %d errors\n' % (time.time() - t0, line_number, errors), file=sys.stderr)

def gft_predict_hf(args):
    global labels
    labels = labels_from_args(args)
    print('labels: ' + str(labels), file=sys.stderr)
    gft_predict_hf_with_pipeline(args)
