#!/usr/bin/env python

import gft
import sys
from gft.gft_internals import my_auto_model_pd
from gft.gft_internals.gft_util import parse_model_specification,parse_dataset_specification, get_arg
from gft.gft_internals import my_datasets
from gft.gft_internals import parse_eqn


# print('loading gft_summary_pd.py', file=sys.stderr)

def get_labels_pd(args):
    model,tokenizer,extractor = my_auto_model_pd.my_load_model_tokenizer_and_extractor_pd(args, 'model')

canconicalized_pipeline_tags = [["AudioClassification"],
                                ["Audio_to_Audio"],
                                ["AutomaticSpeechRecognition", "ctc", "ASR", "automatic-speech-recognition"],
                                ["Conversational"],
                                ["FeatureExtraction"],
                                ["Fill_Mask", "fill-mask"],
                                ["ImageClassification", "image-classification"],
                                ["ImageSegmentation"],
                                ["Image_to_Text"],
                                ["ObjectDetection"],
                                ["QuestionAnswering", "QA", "question_answering", "question-answering", "classify_spans"],
                                ["SentenceSimilarity"],
                                ["StructuredDataClassification"],
                                ["Summarization"],
                                ["TableQuestionAnswering"],
                                ["TextClassification", "text-classification", "classify", "text_classification"],
                                ["TextGeneration"],
                                ["Text_to_Image"],
                                ["Text_to_Speech"],
                                ["TokenClassification", "ner", "classify_tokens"],
                                ["Translation", "translation", "MT"],
                                ["VoiceActivityDetection"],
                                ["Zero_ShotClassification"]]


# def canonicalize_task(task):
#     for aliases in canconicalized_pipeline_tags:
#         if task in aliases:
#             # print('canonicalize_task: %s --> %s' % (str(task), str(aliases[0])), file=sys.stderr) 
#             return aliases[0]
#     # print('canonicalize_task: %s --> %s' % (str(task), str(task)), file=sys.stderr)
#     return task

# from huggingface_hub import HfApi, ModelFilter, ModelSearchArguments

# def my_get(obj, attr, default):
#     if hasattr(obj, attr):
#         return getattr(obj, attr)
#     else: return default

# def print_models(models, prefix):
#     print('%s --> %d models' % (prefix, len(models)))
#     # pdb.set_trace()
#     models2 = [(m.modelId, my_get(m, 'downloads', 0), my_get(m, 'likes', 0)) for m in models]
#     sorted_models2 = sorted(models2, key=lambda x: x[1], reverse=True)

#     for id,downloads,likes in sorted_models2:
#         print('\t'.join([prefix, 'model: ' + str(id), 'downloads: ' + str(downloads), 'likes: ' + str(likes)]))

# def infer_models_from_dataset(args):

#     data_provider,data_key = parse_dataset_specification(args)
#     if data_key is None: return None
#     data_key0 = data_key.split(',')[0]

#     model_args = ModelSearchArguments()
#     d = model_args.dataset

#     if not hasattr(d, data_key0):
#         print('# dataset: %s --> 0 models' % (data_key0))
#     else:
#         d = getattr(d, data_key0)
#         api = HfApi()
#         models = api.list_models(filter=(d))
#         print_models(models, 'dataset: ' + data_key0)
#         # print('# dataset: %s --> %d models' % (data_key0, len(models)))
#         # for m in models:
#         #     print('\t'.join(['dataset: ' + data_key0, 'model: ' + str(m.modelId)]))

# def infer_models_from_task(args):
#     from my_task import infer_task
#     task = canonicalize_task(infer_task(args))

#     model_args = ModelSearchArguments()
#     d = model_args.pipeline_tag
#     if not hasattr(d, task):
#         print('# task: %s --> 0 models' % task)
#     else:
#         d = getattr(d, task)
#         api = HfApi()
#         models = api.list_models(filter=(d))
#         print_models(models, 'task: ' + task)

def summarize_model_pd(args):

    model_provider,model_key = parse_model_specification(args)

    if not model_key is None and not '__infer__' in model_key and not '__contains__' in model_key:
        model,tokenizer,extractor = my_auto_model_pd.my_load_model_tokenizer_and_extractor_pd(args, 'model')

        labels = 'NA'
        name = 'NA'
        nclass = 'NA'

        if hasattr(model, 'get_labels'):
            labels = ', '.join(map(str, model.get_labels()))

        if hasattr(model, 'full_name'):
            name = str(model.full_name())

        elif hasattr(model, 'num_classes'):
            nclass = str(model.num_classes)

        print('\t'.join(['name: ' + name, 'labels (%s): %s' % (nclass, labels), str(model)])) 


def summarize_splits_pd(data_key, datasets):
    if not datasets is None:
        cols = ['%s: %d rows' % (s, len(datasets[s])) for s in datasets]
        print('\t'.join(map(str, ['dataset: ' + data_key, 'splits: ' + ', '.join(cols)])))

def summarize_dataset_pd(args, config=None):

    data_provider,data_key = parse_dataset_specification(args)
    split = get_arg(args, 'split')

    from gft.gft_internals.paddle_dataset_zoo import paddle_dataset_infos
    # print('data_key: ' + str(data_key), file=sys.stderr)
    infos = paddle_dataset_infos(data_key, config=config)
    # print('infos: ' + str(infos), file=sys.stderr)
    # print('config: ' + str(config), file=sys.stderr)

    if config is None:
        for info in infos:
            configs = info['configs']
            # print('configs: ' + str(configs), file=sys.stderr)
            if len(configs) == 0:
                summarize_dataset_pd(args, config='Default')
            else:
                for config in configs:
                    # print('config: ' + str(config), file=sys.stderr)
                    summarize_dataset_pd(args, config=config)
        return

    def hist_col(ds, col):
        res = {}
        for record in ds:
            if col in record:
                v = record[col]
                if not v in res:
                    res[v]=0
                res[v] = res[v] + 1
        res2 = [(k, res[k]) for k in res]
        return sorted(res2, key=lambda p: p[1], reverse=True)

    def summarize_col(ds, col):
        return ', '.join(['%s:%d' % p for p in hist_col(ds, col)])

    if not data_key is None and not '__infer__' in data_key and not '__contains__' in data_key:
        for info in infos:
            if 'splits' in info:
                splits = ','.join(info['splits'])
            else:
                splits = None
            print('info: ' + str(info), file=sys.stderr)
            print('splits: ' + str(splits), file=sys.stderr)
            print('config: ' + str(config), file=sys.stderr)
            datasets = my_datasets.my_load_dataset(args, config=config, default_splits=splits)
            ds = None
            # pdb.set_trace()
            summarize_splits_pd(data_key, datasets)
            if split in datasets: 
                ds = datasets[split]
                print('\t'.join(['dataset: ' + data_key, 'split: ' + split, 'config: ' + str(config), 'cols: ' + ', '.join([key for key in ds[0]])]))
            if split in datasets and hasattr(datasets[split], 'label_list') and not datasets[split].label_list is None:
                print('dataset: ' + str(data_key) + '\tlabels: ' + ', '.join(map(str, datasets[split].label_list)))
            elif split in datasets and not get_arg(args, 'eqn', default=None) is None:
                ds = datasets[split]
                eqn = parse_eqn.parse_eqn(get_arg(args, 'eqn'))
                for col in eqn['y_field_names']:
                    print('\t'.join(map(str, ['dataset: ' + data_key, 'split: ' + split, 'config: ' + str(config), 'col: ' + col, 'labels: ' + summarize_col(ds, col)])))
                for col in eqn['x_field_names']:
                    print('\t'.join(map(str, ['dataset: ' + data_key, 'split: ' + split, 'config: ' + str(config), 'col: ' + col, 'labels: ' + summarize_col(ds, col)])))

    # if not data_key is None and data_key != '__infer__':
    #     datasets = my_datasets.my_load_dataset(args)
    #     ds = None
    #     summarize_splits(data_key, datasets)
    #     if split in datasets: 
    #         ds = datasets[split]
    #         print('\t'.join(['dataset: ' + data_key, 'split: ' + split, 'columns: ' + ', '.join([key for key in ds[0]])]))
    #     if split in datasets and hasattr(datasets[split], 'features'):
    #         feat = datasets[split].features
    #         if 'label' in feat and hasattr(feat['label'], 'names'):
    #             print('dataset: ' + str(data_key) + '\tlabels: ' + ', '.join(map(str, feat['label'].names)))
    #         elif split in datasets and not get_arg(args, 'eqn', default=None) is None:
    #             ds = datsets[split]
    #             eqn = parse_eqn.parse_eqn(get_arg(args, 'eqn'))
    #             for col in eqn['y_field_names']:
    #                 print('\t'.join(map(str, ['dataset: ' + data_key, 'split: ' + split, 'col: ' + col, 'labels: ' + summarize_col(ds, col)])))
    #             for col in eqn['x_field_names']:
    #                 print('\t'.join(map(str, ['dataset: ' + data_key, 'split: ' + split, 'col: ' + col, 'labels: ' + summarize_col(ds, col)])))


def infer_models_from_substring_pd(args, s):
    from gft.gft_internals.paddle_model_zoo import paddle_model_zoo
    matches = [m for m in paddle_model_zoo if s in m.lower()]
    print('\n'.join(matches))

def infer_datasets_from_substring_pd(args, s):
    from gft.gft_internals.paddle_dataset_zoo import paddle_datasets
    for d in paddle_datasets:
        if s in d['name'].lower():
            ss = cc = ''
            if len(d['configs']) > 0:
                cc='configs: ' + '|'.join(d['configs'])
            ss='splits: ' + '|'.join(d['splits'])
            print('\t'.join([d['name'], cc, ss]))

def gft_summary_pd(args):

    summarize_model_pd(args)
    summarize_dataset_pd(args)

    model_provider,model_key = parse_model_specification(args)
    data_provider,data_key = parse_dataset_specification(args)

    if not model_key is None and model_key.startswith('__contains__'):
        infer_models_from_substring_pd(args, model_key[len('__contains__'):].lower())

    if not data_key is None and  data_key.startswith('__contains__'):
        infer_datasets_from_substring_pd(args, data_key[len('__contains__'):].lower())

    
    if not model_key is None:
        assert not '__infer__' in model_key, '__infer__ is not supported for Paddle (yet)'

    if not data_key is None:
        assert not '__infer__' in data_key, '__infer__ is not supported for Paddle (yet)'

