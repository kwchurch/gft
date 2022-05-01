#!/usr/bin/env python

import gft
from gft.gft_internals import my_auto_model_hf
from gft.gft_internals.gft_util import parse_model_specification,parse_dataset_specification, get_arg
from gft.gft_internals import my_datasets
from gft.gft_internals import parse_eqn


# def get_labels(args):
#     model,tokenizer,extractor = my_auto_model_hf.my_load_model_tokenizer_and_extractor(args, 'model')

from huggingface_hub import HfApi, ModelSearchArguments, list_models, list_datasets, model_info, dataset_info

def my_get(obj, attr, default):
    if hasattr(obj, attr):
        return getattr(obj, attr)
    else: return default

def dict2str(d):
    return '\t'.join([str(k) + ': ' + str(d[k]) for k in d])

def print_models(models, prefix, print_number=True):
    if print_number: print('%s --> %d models' % (prefix, len(models)))
    # tags = ', '.join(my_get(m, 'tags', []))
    models2 = [{'model' : m.modelId, 
                'downloads' : my_get(m, 'downloads', 0), 
                'likes': my_get(m, 'likes', 0), 
                'task' : my_get(m, 'pipeline_tag', '')}
               for m in models]
    sorted_models2 = sorted(models2, key=lambda x: x['downloads'], reverse=True)

    for d in sorted_models2:
        print(prefix + '\t' + dict2str(d))


def print_datasets(datasets, prefix, print_number=True):
    if print_number: print('%s --> %d datasets' % (prefix, len(datasets)))

    # tags = ', '.join(my_get(d, 'tags', []))
    datasets2 = [(d.id, my_get(d, 'downloads', 0), my_get(d, 'likes', 0), my_get(d, 'paperswithcode_id', None)) for d in datasets]
    sorted_downloads2 = sorted(datasets2, key=lambda x: x[1], reverse=True)

    for id,downloads,likes,pwc in sorted_downloads2:
        if pwc is None: pwc = ''
        else: pwc = 'PWC: https://paperswithcode.com/dataset/' + pwc
        print('\t'.join([prefix, 
                         'dataset: ' + str(id), 
                         'downloads: ' + str(downloads), 
                         'likes: ' + str(likes),
                         pwc]))

# given a dataset, generate a list of models, sorted by downloads
def infer_models_from_dataset(args):

    data_provider,data_key = parse_dataset_specification(args)

    if data_key is None: return
    if '__contains__' in data_key: return

    data_key0 = data_key.split(',')[0]

    try:
        model_args = ModelSearchArguments()
    except: 
        import sys
        print('Warning: HuggingFace ModelSearchArguments is not working', file=sys.stderr)
        return

    d = model_args.dataset

    if not hasattr(d, data_key0):
        print('# dataset: %s --> 0 models' % (data_key0))
    else:
        d = getattr(d, data_key0)
        api = HfApi()
        models = api.list_models(filter=(d))
        topn = get_arg(args, 'topn')
        if len(models) > topn: models = models[0:topn]
        print_models(models, 'dataset: ' + data_key0)

def apply_model_info(models):
    try:
        return [model_info(m.modelId) for m in models]
    except:
        return models

# given a task, generate a list of models, sorted by downloads
def infer_models_from_task(args):
    from gft.gft_internals.my_task import infer_task,canonicalize_task
    task = canonicalize_task(infer_task(args))
    # if task is None: return
    topn = get_arg(args, 'topn')

    try:
        model_args = ModelSearchArguments()
    except: 
        import sys
        print('Warning: HuggingFace ModelSearchArguments is not working', file=sys.stderr)
        return

    d = model_args.pipeline_tag
    # print('task: ' + task)
    if not hasattr(d, str(task)):
        # print('# task: %s --> 0 models' % task)
        models = [m for m in list_models() if hasattr(m, 'pipeline_tag') and m.pipeline_tag == task ]
        if len(models) > topn: 
            print('# found: %d matches; truncating to %d' % (len(models), topn))
            models = models[0:topn]
        if not get_arg(args, 'fast_mode', default=False):
            models = apply_model_info(models)
        print_models(models, 'task: ' + str(task))
    else:
        d = getattr(d, task)
        api = HfApi()
        models = api.list_models(filter=(d))
        topn = get_arg(args, 'topn')
        if len(models) > topn: models = models[0:topn]
        print_models(models, 'task: ' + task)

# huggingface_hub.model_info(models[0].modelId)

# def infer_models_from_substring(args, s):
#     print('s: ' + str(s))
#     api = HfApi()
#     models = api.list_models(filter=(s))
#     print_models(models, 'substring: ' + s)


# model info is slow
# cannot afford to run this on 30k models
# it appears that the list is already sorted by downloads
def infer_models_from_substring(args, s):
    topn = get_arg(args, 'topn')

    matches = [m for m in list_models() if s in str(m.modelId).lower()]
    if len(matches) > topn:
        print('# found: %d matches; truncating to %d' % (len(matches), topn))
        matches = matches[0:topn]

    models = matches
    if not get_arg(args, 'fast_mode', default=False):
        models = apply_model_info(matches)
    # models = [model_info(m.modelId) for m in matches]
    print_models(models, 'substring: ' + s)

def infer_datasets_from_substring(args, s):
    topn = get_arg(args, 'topn')

    matches = [d for d in list_datasets() if s in str(d.id).lower()]
    if len(matches) > topn:
        print('# found: %d matches; truncating to %d' % (len(matches), topn))
        matches = sorted(matches, key=lambda x: my_get(x, 'downloads', 0), reverse=True)[0:topn]

    datasets = matches
    if not get_arg(args, 'fast_mode', default=False):
        datasets = [dataset_info(d.id) for d in matches]
    print_datasets(datasets, 'substring: ' + s)

def summarize_model(args):

    model_provider,model_key = parse_model_specification(args, keyword='model')

    if model_key is None or '__contains__' in model_key or '__infer__' in model_key: return

    prefix = 'model: ' + str(model_key)

    try:
        m = model_info(model_key)
        print_models([m], prefix, print_number=False)
    except:
        print(prefix + '\tNo info from HuggingFace')

    model,tokenizer,extractor = my_auto_model_hf.my_load_model_tokenizer_and_extractor(args, 'model')

    from gft.gft_internals.gft_util import labels_from_model
    labs = labels_from_model(model)
    if not labs is None:
        print('\t'.join(map(str, [prefix, 'labels: ' +  ', '.join(map(str, labs))])))
    else:
        print('\t'.join(map(str, [prefix, 'labels: NA'])))

    print(model)
    
    # try:
    #     if hasattr(model, 'get_labels'):
    #         labs = model.get_labels()
    #         if not labs is None:
    #             print('\t'.join(map(str, [prefix, 'labels: ' +  ', '.join(map(str, labs))])))
    #     elif hasattr(model, 'config'):        
    #         print('\t'.join(map(str, [prefix, 'labels: ' + ', '.join(map(str, model.config.label2id.keys()))])))
    #     else:
    #         print('\t'.join(map(str, [prefix, 'labels: NA'])))
    # except:
    #     print('\t'.join(map(str, [prefix, 'labels: NA'])))
        
def summarize_splits(data_key, datasets):
    if not datasets is None:
        cols = ['%s: %d rows' % (s, len(datasets[s])) for s in datasets]
        print('\t'.join(map(str, ['dataset: ' + data_key, 'splits: ' + ', '.join(cols)])))

def summarize_dataset(args):

    data_provider,data_key = parse_dataset_specification(args)

    if data_key is None or '__contains__' in data_key or '__infer__' in data_key: return

    try:
        if ',' in data_key:
            d0 = data_key.split(',')[0]
        else:
            d0 = data_key
        prefix = 'dataset: ' + str(d0)
        d = dataset_info(d0)
        print_datasets([d], prefix, print_number=False)
    except:
        print(prefix + '\tNo info from HuggingFace')

    if ',' in data_key:
        return summarize_dataset_internal(data_key, args)

    from datasets import get_dataset_config_names

    try:
        configs = get_dataset_config_names(data_key)
    except:
        configs = ['default']

    if configs == ['default']:
        return summarize_dataset_internal(data_key, args)
    for config in configs:
        summarize_dataset_internal(data_key + ',' + config, args)

def summarize_dataset_internal(data_key, args):
    prefix = 'dataset: ' + str(data_key)
    split = get_arg(args, 'split')

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

    if not data_key is None and data_key != '__infer__':
        datasets = my_datasets.my_load_dataset(args, data_key=data_key)
        ds = None
        summarize_splits(data_key, datasets)
        if split in datasets: 
            ds = datasets[split]
            try:
                keys = [key for key in ds[0]]
            except:
                keys = []
            print('\t'.join(['dataset: ' + data_key, 'split: ' + split, 'columns: ' + ', '.join(keys)]))
        if split in datasets and hasattr(datasets[split], 'features'):
            feat = datasets[split].features
            if 'label' in feat and hasattr(feat['label'], 'names'):
                print('dataset: ' + str(data_key) + '\tlabels: ' + ', '.join(map(str, feat['label'].names)))
            elif split in datasets and not get_arg(args, 'eqn', default=None) is None:
                ds = datasets[split]
                eqn = parse_eqn.parse_eqn(get_arg(args, 'eqn'))
                for col in eqn['y_field_names']:
                    print('\t'.join(map(str, ['dataset: ' + data_key, 'split: ' + split, 'col: ' + col, 'labels: ' + summarize_col(ds, col)])))
                for col in eqn['x_field_names']:
                    print('\t'.join(map(str, ['dataset: ' + data_key, 'split: ' + split, 'col: ' + col, 'labels: ' + summarize_col(ds, col)])))


def gft_summary_hf(args):
    
    model_provider,model_key = parse_model_specification(args, keyword='model')
    data_provider,data_key = parse_dataset_specification(args)
    
    summarize_model(args)
    summarize_dataset(args)

    from gft.gft_internals.my_task import summarize_task
    summarize_task(args)

    if model_key == '__infer__' or model_key is None:
        infer_models_from_dataset(args)
        infer_models_from_task(args)

    if not model_key is None and model_key.startswith('__contains__'):
        infer_models_from_substring(args, model_key[len('__contains__'):].lower())

    if not data_key is None and  data_key.startswith('__contains__'):
        infer_datasets_from_substring(args, data_key[len('__contains__'):].lower())
