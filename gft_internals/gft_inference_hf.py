# convention: _hf is from HuggingFace, and _pd is from PaddleHub

import numpy as np
import os,sys,json,time
import torch

print(__name__ + ': loading from huggingface', file=sys.stderr)

t0 = time.time()

from gft_internals.gft_util import parse_model_specification,parse_dataset_specification,parse_task_specification,get_arg
from gft_internals import my_datasets

from gft_internals import my_auto_model_hf

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
    if labels is None: return str(np.argmax(logits))
    else: return labels[np.argmax(logits)]

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
    
    if get_arg(args, 'debug', default=False):
        return apply_pipeline_internal(args, xfields, pipe, task, model, tokenizer), 0
    else:
        try: return apply_pipeline_internal(args, xfields, pipe, task, model, tokenizer), 0
        except: return '***ERROR***', 1

def apply_pipeline_internal(args, xfields, pipe, task, model, tokenizer):

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
        res = pipe(xfields[0])
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
        assert False, 'apply_pipeline_internal, task not supported: ' + task

    if task == "text-classification" or task == "sentiment-analysis":
        texts = [xfields[0]]
        res = pipe(texts)
        return str(res[0]['label']) + '\t' + str(res[0]['score'])

    if task == "text-generation":
        res = pipe(xfields[0], max_length=get_arg(args, 'max_length', default=30), num_return_sequences=get_arg(args, 'num_return_sequences', default=3))
        return '\t'.join([r['generated_text'] for r in res ])

    if task == "token-classification" or task == "ner":
        res = pipe(xfields[0])
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

def gft_inference_hf_with_pipeline(args):
    from transformers import pipeline
    model_provider,model_key = parse_model_specification(args, keyword='model')
    assert model_provider != 'PaddleNLP', 'expected provider to be HuggingFace; provider: ' + str(provider)

    delim = get_arg(args, 'delimiter')
    task_provider,task = parse_task_specification(args)
    pipe = None
    model = None
    tokenizer = None
    extractor = None

    model,tokenizer,extractor = my_auto_model_hf.my_load_model_tokenizer_and_extractor(args, 'model')
    if not model is None and tokenizer is None:
        from transformers import AutoFeatureExtractor
        extractor = AutoFeatureExtractor.from_pretrained(model_key)

    if task is None:
        pipe = None
    elif model is None:
        pipe = pipeline(task)
    elif tokenizer is None:
        assert not extractor is None, 'confusion in gft_inference_hf_with_pipeline'
        pipe = pipeline(task, model=model, feature_extractor=extractor)
    else:
        pipe = pipeline(task, model=model, tokenizer=tokenizer)
        
    line_number=0
    errors=0
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

    print('total time: %0.2f seconds; processed %d input lines; caught %d errors\n' % (time.time() - t0, line_number, errors), file=sys.stderr)

def gft_inference_hf(args):
    global labels
    labels = labels_from_args(args)
    eqn = 
    gft_inference_hf_with_pipeline(args)
