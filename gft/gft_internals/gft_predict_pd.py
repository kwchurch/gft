# convention: _hf is from HuggingFace, and _pd is from PaddleHub

import gft
from paddlenlp import Taskflow

# https://paddlenlp.readthedocs.io/en/stable/model_zoo/taskflow.html
# # 中文分词
# seg = Taskflow("word_segmentation")
# seg("第十四届全运会在西安举办")
# >>> ['第十四届', '全运会', '在', '西安', '举办']

# # 词性标注
# tag = Taskflow("pos_tagging")
# tag("第十四届全运会在西安举办")
# >>> [('第十四届', 'm'), ('全运会', 'nz'), ('在', 'p'), ('西安', 'LOC'), ('举办', 'v')]

# # 命名实体识别
# ner = Taskflow("ner")
# ner("《孤女》是2010年九州出版社出版的小说，作者是余兼羽")
# >>> [('《', 'w'), ('孤女', '作品类_实体'), ('》', 'w'), ('是', '肯定词'), ('2010年', '时间类'), ('九州出版社', '组织机构类'), ('出版', '场景事件'), ('的', '助词'), ('小说', '作品类_概念'), ('，', 'w'), ('作者', '人物类_概念'), ('是', '肯定词'), ('余兼羽', '人物类_实体')]

# # 『解语』- 名词短语标注
# nptag = Taskflow("knowledge_mining", model="nptag")
# nptag("红曲霉菌")
# >>> [{'text': '红曲霉菌', 'label': '微生物'}]

# # 句法分析
# ddp = Taskflow("dependency_parsing")
# ddp("9月9日上午纳达尔在亚瑟·阿什球场击败俄罗斯球员梅德韦杰夫")
# >>> [{'word': ['9月9日', '上午', '纳达尔', '在', '亚瑟·阿什球场', '击败', '俄罗斯', '球员', '梅德韦杰夫'], 'head': [2, 6, 6, 5, 6, 0, 8, 9, 6], 'deprel': ['ATT', 'ADV', 'SBV', 'MT', 'ADV', 'HED', 'ATT', 'ATT', 'VOB']}]

# # 情感分析
# senta = Taskflow("sentiment_analysis")
# senta("这个产品用起来真的很流畅，我非常喜欢")
# >>> [{'text': '这个产品用起来真的很流畅，我非常喜欢', 'label': 'positive', 'score': 0.9938690066337585}]

import time
t0 = time.time()

import numpy as np
import os,sys,json
import torch

print(__name__ + ': loading from paddle', file=sys.stderr)

from gft.gft_internals.gft_util import parse_model_specification,parse_dataset_specification,parse_task_specification,get_arg
from gft.gft_internals import my_datasets
# from gft_util import parse_model_specification,get_arg

import paddle
from paddlenlp.transformers import (
    AutoModelForSequenceClassification as AutoModelForSequenceClassification_pd,
    AutoTokenizer as AutoTokenizer_pd,
)

from paddle.io import DataLoader as DataLoader_pd

def f2s(f):
    return str(f)

def floats2str(fs):
    return '|'.join(map(str, fs))

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

# def old_gft_predict_pd(args):
#     print('calling gft_predict_pd with args: ' + str(args))
#     provider,model_key = parse_model_specification(args, keyword='model')

#     assert provider == 'PaddleHub', 'expected provider to be PaddleHub; provider: ' + str(provider)

#     labels = get_arg(args, 'labels', default=None)

#     if labels is None:
#         p = os.path.join(model_key + 'labels')
#         if os.path.exists(p): labels = p

#     if not labels is None:
#         with open(labels, 'r') as fd:
#             labels = fd.read().split('\n')

#     def best_label(logits):
#         if labels is None:
#             return str(np.argmax(logits))
#         else:
#             return labels[np.argmax(logits)]

#     model = AutoModelForSequenceClassification_pd.from_pretrained(model_key)
#     tokenizer = AutoTokenizer_pd.from_pretrained(model_key)

#     delim = get_arg(args, 'delimiter')

#     # def old_my_apply(X):
#     #     tokenized_input = tokenizer(*X)
#     #     input_ids = tokenized_input['input_ids']
#     #     token_type_ids = tokenized_input['token_type_ids']
#     #     tokens = tokenizer.convert_ids_to_tokens(input_ids)
#     #     return model(paddle.to_tensor([input_ids]), token_type_ids=paddle.to_tensor([token_type_ids]))

#     # def my_apply(X):
#     #     return model(paddle.to_tensor([tokenizer(*X)['input_ids']]))

#     line_number=0
#     errors=0
#     for line in sys.stdin:
#         line_number +=1
#         fields = line.rstrip().split(delim)
#         xfields = None

#         if len(fields) == 0: continue

#         if len(fields) >= 1: 
#             xfields = fields[0].split('|')

#         try:
#             outputs = model(paddle.to_tensor([tokenizer(*xfields)['input_ids']]))
#             logits = outputs.detach().numpy()[0]
#             print(delim.join([line.rstrip(), floats2str(logits), best_label(logits)]))
#             sys.stdout.flush()
#         except:
#             print(delim.join([line.rstrip(), '***ERROR***']))
#             errors += 1

#     print('total time: %0.2f seconds; processed %d input lines; caught %d errors' % (time.time() - t0, line_number, errors), file=sys.stderr)

def apply_task(args, xfields, tf, task, model, tokenizer):
    assert not tokenizer is None, 'tokenizer should not be None'

    if get_arg(args, 'do_not_catch_errors', default=False):
        return apply_task_internal(args, xfields, tf, task, model, tokenizer), 0
    else:
        try: return apply_task_internal(args, xfields, tf, task, model, tokenizer), 0
        except: return '***ERROR***', 1

# current set of Taskflow tasks:
# knowledge_mining, ner, poetry_generation, question_answering, lexical_analysis, word_segmentation, 
# pos_tagging, sentiment_analysis, dependency_parsing, text_correction, text_similarity, dialogue

def apply_task_internal(args, xfields, tf, task, model, tokenizer):

    print('apply_task_internal: task = ' + str(task) + ' tf = ' + str(tf), file=sys.stderr)
    print('xfields: ' + str(xfields), file=sys.stderr)
    assert not tokenizer is None, 'tokenizer should not be None'

    if task is None:
        assert not model is None, 'model should not be None'
        assert not tokenizer is None, 'tokenizer should not be None'
        tokens = tokenizer(*xfields)
        print('tokens: ' + str(tokens), file=sys.stderr)
        # outputs = model(paddle.to_tensor([tokens]))
        outputs = model(paddle.to_tensor([tokens['input_ids']]),
                        token_type_ids=paddle.to_tensor([tokens['token_type_ids']]))


        print('outputs: ' + str(outputs), file=sys.stderr)
        # import pdb
        # pdb.set_trace()

        # logits = outputs.detach().numpy()[0]
        logits = outputs[1].numpy()[:,0][0]
        return str(logits)

        # I really don't understand PaddleNLP models
        # return best_label(logits) + '\t' + floats2str(logits)

    print('xfields: ' + str(xfields), file=sys.stderr)
    res = tf(*xfields)
    print('res: ' + str(res), file=sys.stderr)

    if task == 'word_segmentation':
        return ' '.join(tf(*xfields))
    if task == 'pos_tagging' or task == 'ner':
        res = tf(*xfields)
        return ' '.join([word + '/' + tag for word,tag in res])

    # sentiment_analysis
    if 'label' in res[0] and 'score' in res[0]: 
        return '\t'.join(['%s\t%0.3f' % (r['label'], r['score']) for r in res])

    for keyword in ['label', 'answer', 'similarity', 'target']:
        if keyword in res[0]: return '\t'.join(map(str, [r[keyword] for r in res]))

    if 'items' in res[0]: return '\t'.join(['%s/%s' % (item['item'], item['wordtag_label']) for item in res[0]['items']])

    # lexical_analyssis
    if 'segs' in res[0] and 'tags' in res[0]:
        return '\t'.join(['%s/%s' % (seg, tag) for seg,tag in zip(res[0]['segs'],res[0]['tags'])])

    # dependency_parsing
    if 'word' in res[0] and 'head' in res[0] and 'deprel' in res[0]:
        return '\t'.join(['%d:%s/%s/%s' % (i,word,head,dep) for i,(word,head,dep) in enumerate(zip(res[0]['word'],res[0]['head'],res[0]['deprel']))])

    # if task == 'dependency_parsing':
    #     assert False, 'apply_task_internal, task not supported: ' + task
    if task == 'text_correction':
        assert False, 'apply_task_internal, task not supported: ' + task

    if task == 'text_similarity':
        res = tf(*xfields)
        return '\t'.join(map(str, [r['similarity'] for r in res ]))

    if task == 'dialogue':
        assert False, 'apply_task_internal, task not supported: ' + task

    if task == "audio-classification":
        assert False, 'apply_pipeline_internal, task not supported: ' + task

    if task == "automatic-speech-recognition":
        assert False, 'apply_pipeline_internal, task not supported: ' + task
        # res = pipe(xfields[0])
        # return res['text']

    if task == "conversational":
        assert False, 'apply_pipeline_internal, task not supported: ' + task

    if task == "feature-extraction":
        assert False, 'apply_pipeline_internal, task not supported: ' + task

    if task == "fill-mask":
        assert False, 'apply_pipeline_internal, task not supported: ' + task
        # res = pipe(xfields[0])
        # return '\t'.join(['%s|%0.3f' % (r['token_str'], r['score']) for r in res ])

    if task == "image-classification":
        assert False, 'apply_pipeline_internal, task not supported: ' + task
        # res = pipe(xfields[0])
        # return '\t'.join(['%s|%0.3f' % (r['label'], r['score']) for r in res ])

    if task == "question-answering":
        assert False, 'apply_pipeline_internal, task not supported: ' + task
        # assert len(xfields) == 2, 'expected 2 fields: found ' + str(len(xfields))
        # question,context=xfields[0:2]
        # res = pipe({'question' : question, 'context' : context})
        # return 'answer: ' + res['answer'] + '\tscore: %0.4f' % res['score'] + ' span: %d-%d' % (res['start'], res['end'])

    if task == "table-question-answering":
        assert False, 'apply_pipeline_internal, task not supported: ' + task

    if task == "text2text-generation":
        assert False, 'apply_pipeline_internal, task not supported: ' + task

    if task == "text-generation":
        assert False, 'apply_pipeline_internal, task not supported: ' + task
        # res = pipe(xfields[0], max_length=get_arg(args, 'max_length', default=30), num_return_sequences=get_arg(args, 'num_return_sequences', default=3))
        # return '\t'.join([r['generated_text'] for r in res ])

    if task == "token-classification" or task == "ner":
        assert False, 'apply_pipeline_internal, task not supported: ' + task
        # res = pipe(xfields[0])
        # return '\t'.join(['%s/%s:%0.4f' % (r['word'], r['entity'], r['score']) for r in res])

    if task == "MT" or task == "translation":
        assert False, 'apply_pipeline_internal, task not supported: ' + task
        # res = pipe(*xfields)
        # return res[0]['translation_text']

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

def gft_predict_pd_with_pipeline(args):
    # from transformers import pipeline
    model_provider,model_key = parse_model_specification(args, keyword='model')
    assert model_provider != 'HuggingFace', 'expected provider to be PaddleHub; provider: ' + str(model_provider)

    delim = get_arg(args, 'delimiter')
    provider,task = parse_task_specification(args)

    if task == 'help':
        print('HELP, taskflow: ' + ', '.join(Taskflow.tasks()))
        sys.exit()

    print('task: ' + str(task))

    tf = None
    pipe = None
    model = None
    tokenizer = None
    extractor = None

    if task in Taskflow.tasks():
        if model_key is None:
            tf = Taskflow(task)
        else:
            tf = Taskflow(task, model=model_key)

    from gft.gft_internals import my_auto_model_pd
    model,tokenizer,extractor = my_auto_model_pd.my_load_model_tokenizer_and_extractor_pd(args, 'model')

    # if not model is None and tokenizer is None:
    #     from transformers import AutoFeatureExtractor
    #     extractor = AutoFeatureExtractor.from_pretrained(model_key)

    # if task is None:
    #     pipe = None
    # elif model is None:
    #     pipe = pipeline(task)
    # elif tokenizer is None:
    #     assert not extractor is None, 'confusion in gft_predict_hf_with_pipeline'
    #     pipe = pipeline(task, model=model, feature_extractor=extractor)
    # else:
    #     pipe = pipeline(task, model=model, tokenizer=tokenizer)
        
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

            # res,err = apply_pipeline(args, xfields, pipe, task, model, tokenizer)
            res,err = apply_task(args, xfields, tf, task, model, tokenizer)
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
            res,err = apply_task(args, xfields, tf, task, model, tokenizer)
            errors += err
            print(delim.join(['|'.join(map(str, xfields)), 
                              '|'.join(map(str, yfields)), 
                              res]))
            sys.stdout.flush()
        

    print('total time: %0.2f seconds; processed %d input lines; caught %d errors\n' % (time.time() - t0, line_number, errors), file=sys.stderr)

def gft_predict_pd(args):
    global labels
    labels = labels_from_args(args)
    gft_predict_pd_with_pipeline(args)
