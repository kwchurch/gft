# from datasets import ClassLabel
import gft
import os, sys

# args is a dict of dicts and/or classes
# examples of classes: HuggingFace TrainingArguments
# classes are returned by HfArgumentParser in HuggingFace (and argparse, more generally)

# get_arg and set_arg search args
# it is assumed that there is only one match

def get_arg(args, arg, default='***SIGNAL_ERROR***'):
    if 'extras' in args and arg in args['extras']:
        return args['extras'][arg]
    for key in args:
        if hasattr(args[key], arg):
            return getattr(args[key], arg)
    if default == '***SIGNAL_ERROR***':
        assert False, 'get_arg: cannot find ' + str(arg)
    else: return default

def set_arg(args, arg, val):
    print('arg: %s --> %s' % (arg, val), file=sys.stderr)
    for key in args:
        if hasattr(args[key], arg):
            setattr(args[key], arg,val)
            return None
    if not 'extras' in args:
        args['extras'] = {}

    args['extras'][arg] = val
    # assert False, 'set_arg: cannot find ' + str(arg)

# may need to distinguish these various Paddle providers at some point in the future

providers = [['HuggingFace', 'H'], 
             ['PaddleHub', 'PaddleNLP', 'PaddlePaddle', 'Paddle', 'P'], 
             ['Custom', 'C']]

normalized_providers = {}
for p in providers:
    for nickname in p:
        normalized_providers[nickname] = p[0]


# model specifications: 
#	<provider> : model_key
# Model_key can be anything supported by hubs such as
# https://huggingface.co/models
# https://github.com/PaddlePaddle/PaddleNLP/blob/develop/docs/model_zoo/transformers.rst
# Custom models are also supported.  In that case, the model_key should be a pathname: p
# p.train, p.val and p.test should be csv files on the local filesystem.

def parse_supplier_prefix(s):
    if s is None: return normalized_providers["C"], None
    fields = s.split(':')

    if len(fields) == 1:
        return normalized_providers["H"], fields[0]

    prefix = fields[0]

    assert prefix in normalized_providers, 'parse_supplier_prefix, bad arg: ' + str(s)
    return normalized_providers[prefix], ':'.join(fields[1:])

# current set of Taskflow tasks:
# knowledge_mining, ner, poetry_generation, question_answering, lexical_analysis, word_segmentation, 
# pos_tagging, sentiment_analysis, dependency_parsing, text_correction, text_similarity, dialogue

task_aliases_pd = {}
task_aliases_hf = {}

for row in [['translation', 'MT', 'machine-translation', 'machine_translation'],
            ['ner', 'token-classification', 'token_classification'],
            ['automatic-speech-recognition', 'ASR'],
            ['question_answering', 'QA', 'question-answering'],
            ['sentiment_analysis', 'sentiment-analysis', 'text_classification', 'text-classification']]:
    for alias in row[1:]:
        task_aliases_pd[alias] = row[0]

for row in [['translation', 'MT', 'machine-translation', 'machine_translation'],
            ['automatic-speech-recognition', 'ASR'],
            ['question-answering', 'question_answering', 'QA'],
            ['sentiment_analysis', 'sentiment-analysis', 'text_classification', 'text-classification']]:
    for alias in row[1:]:
        task_aliases_hf[alias] = row[0]
    
def parse_model_specification(args, keyword='model'):
    provider,model = parse_supplier_prefix(get_arg(args, keyword, default=None))
    return provider,model

    # pretrained = get_arg(args, keyword, default=None)
    # if pretrained is None: 
    #     return normalized_providers["C"], None
    # fields = pretrained.split(':')
    # if len(fields) == 1:
    #     return 'HuggingFace', fields[0]
    # assert len(fields) == 2, 'bad --pretrained arg: ' + str(pretrained)
    # provider,key = fields
    # assert provider in normalized_providers, 'bad --pretrained arg: ' + str(pretrained) + '; unknown provider: ' + provider
    # print('parse_model_specification: provider = %s, key = %s' % (normalized_providers[provider], key), file=sys.stderr)
    # return normalized_providers[provider], key


def parse_base_model_specification(args):
    model_provider,model_key = parse_supplier_prefix(get_arg(args, 'model', default=None))
    base_model_provider,base_model_key = parse_supplier_prefix(get_arg(args, 'base_model', default=None))

    def one_of(choices, superstring):
        for choice in choices:
            if choice in superstring:
                return choice
        return choices[-1]

    if not base_model_key is None:
        if '__one_of__' in base_model_key:
            choices = base_model_key[len('__one_of__'):].split(',')
            base_model_key = one_of(choices, model_key)

    return base_model_provider, base_model_key, model_provider, model_key

def parse_metric_specification(args, keyword='metric'):
    return parse_supplier_prefix(get_arg(args, keyword, default=None))

    # metric = get_arg(args, 'metric', default=None)
    # if metric is None: return None,None
    # fields = metric.split(':')
    # if len(fields) == 1:
    #     return 'HuggingFace', fields[0]
    # assert len(fields) == 2, 'bad --metric arg: ' + str(metric)
    # provider,key = fields
    # assert provider in normalized_providers, 'bad --metric arg: ' + str(metric) + '; unknown provider: ' + provider
    # print('parse_metic_specification: provider = %s, key = %s' % (normalized_providers[provider], key), file=sys.stderr)
    # return normalized_providers[provider], key

def parse_dataset_specification(args, keyword='data'):
    return parse_supplier_prefix(get_arg(args, keyword, default=None))

    # data = get_arg(args, 'data', default=None)
    # if data is None: return data,data
    # fields = data.split(':')
    # provider = fields[0]
    # if len(fields) < 2:
    #     assert 'parse_dataset_specifier: cannot parse ' + data
    # rhs = data[data.find(':') + 1:]
    # assert provider in normalized_providers, 'parse_dataset_specifier: unexpected provider = ' + str(provider)
    # print('parse_dataset_specification: provider = %s, rhs = %s' % (normalized_providers[provider], rhs), file=sys.stderr)
    # return normalized_providers[provider], rhs    

def old_parse_model_specification(args, keyword='task'):
    provider,task = parse_supplier_prefix(get_arg(args, keyword, default=None))

    if provider == 'PaddleHub': task_aliases = task_aliases_pd
    else: task_aliases = task_aliases_hf

    if task in task_aliases: task = task_aliases[task]
    return provider,task

def parse_task_specification(args, keyword='task'):
    return parse_supplier_prefix(get_arg(args, keyword, default=None))

def my_split(val):
    if isinstance(val, str): return val.split()
    else: return [val]

def labels_from_model(model):
    if hasattr(model, 'get_labels'):
        labs = model.get_labels()
        if not labs is None:
            return labs
    if hasattr(model, 'config'):        
        return model.config.label2id.keys()
    return None

def invert_labels(labels):
    return {l: i for i, l in enumerate(labels)}

def intern_labels(dataset, fields, args, labs=None):
    if not labs is None:
        return invert_labels(labs),labs
    elif hasattr(dataset['train'], 'features') and hasattr(dataset['train'].features[fields[0]], 'names'):
        labels = dataset['train'].features[fields[0]].names
        print('intern_labels: ' + str(labels), file=sys.stderr)
        # res = {l: i for i, l in enumerate(labels)}
        return invert_labels(labels),labels
    elif hasattr(dataset['train'], 'label_list') and not dataset['train'].label_list is None:
        labels = dataset['train'].label_list
        print('intern_labels: ' + str(labels), file=sys.stderr)
        # res = {l: i for i, l in enumerate(labels)}
        return invert_labels(labels),labels
    else:
        print('intern_labels: computing defaults', file=sys.stderr)
        res = {}
        for split in dataset:
            if split == 'test': continue
            for record in dataset[split]:
                for field in fields:
                    if field in record:
                        val = record[field]
                        if isinstance(val, list):
                            for v in val:
                                if not v in res:
                                    res[v] = len(res)
                        else:
                            for v in my_split(val):
                                if not v in res:
                                    res[v] = len(res)
        labels = sorted(res.keys()) # sort for determinism 
        sorted_res = {l: i for i, l in enumerate(labels)}
        for i in range(len(labels)):
            sorted_res[i] = i
        print('interned_labels: ' + str(sorted_res), file=sys.stderr)
        assert len(labels) > 0, 'intern_labels: no labels???'
        output_dir = get_arg(args, 'output_dir', default=None)
        # print('output_dir: ' + str(output_dir))
        # import pdb
        # pdb.set_trace()
        if os.path.exists(str(output_dir)):
            with open(os.path.join(str(output_dir), 'labels'), 'w') as fd:
                for lab in sorted_res:
                    print(str(lab) + '\t' + str(sorted_res[lab]), file=fd)
        return sorted_res,labels

def better(args, metric, best_so_far):
    fig = get_arg(args, 'figure_of_merit', default=None)
    better_direction = get_arg(args, 'better_figure_of_merit', default=1)

    if fig is None: 
        fig = 'accuracy'

    if best_so_far is None: 
            return True, metric[fig]

    if better_direction > 0:
        return (metric[fig] > best_so_far), metric[fig]
    else: 
        return (metric[fig] < best_so_far), metric[fig]

def checkpoint_filename(args, model_key, epoch, best):
    checkpoint = get_arg(args, 'output_dir', default=None)
    if checkpoint is None: return None
    suffix = 'best'
    if not best is True:
        suffix = 'epoch.' + str(epoch)
    return '%s/%s' % (checkpoint, suffix)
