
# adapters are really powerful, but unfortunately, they are a bit tricky to use with automodels
# There are also some unfortunate incompatibilities between adpaters and the most recent version of transformers

import gft
import sys,os

from transformers import AutoTokenizer,AutoFeatureExtractor

from gft.gft_internals.gft_util import parse_base_model_specification


def get_config(fn):
    p = os.path.join(fn, 'config.json')
    if os.path.exists(p):
        with open(p, 'r') as fd:
            import json
            return json.loads(fd.read())
    return None

# DeltaHub looks similar to adapters, but more of a work in progress
# installation is harder than pip
# And it seems to be limited to tensorflow
#
# The documentation suggests you can say
#   pip install opendelta
#
# but I had to do this:
# git clone https://github.com/thunlp/OpenDelta.git
# cd OpenDelta
# python setup.py install
#
# According to https://huggingface.co/DeltaHub/lora_t5-base_mrpc,
# you should be able to do the following, but that is generating
# warnings about checksums not matching.
# Maybe opendelta is still too unstable for now.
#
# from transformers import AutoModelForSeq2SeqLM
# t5 = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
# from opendelta import AutoDeltaModel
# delta = AutoDeltaModel.from_finetuned("DeltaHub/lora_t5-base_mrpc", backbone_model=t5)
# delta.log()


def my_get_adapter_info(adapter_key):
    try:
        import transformers.adapters.utils
        a = transformers.adapters.utils.get_adapter_info(adapter_key, 'hf')
        if a is None:
            a = transformers.adapters.utils.get_adapter_info(adapter_key, 'ah')
        # assert not a is None, 'bad adapter key: ' + str(adapter_key)
        return a
    except:
        if adapter_key.startswith('AdapterHub/'):
            import transformers
            assert hasattr(transformers, 'adapters'), 'adapters are not supported in newer version of transformers; tranformers.__version__ is %s; adapter_key is: %s' % (transformers.__version__, adapter_key)
        return None

# from gft_internals.my_auto_model_util import my_tokenizer


# ['FNetForSequenceClassification', 'GPTJForSequenceClassification',
#  'LayoutLMv2ForSequenceClassification',
#  'RemBertForSequenceClassification',
#  'CanineForSequenceClassification',
#  'RoFormerForSequenceClassification',
#  'BigBirdPegasusForSequenceClassification',
#  'BigBirdForSequenceClassification',
#  'ConvBertForSequenceClassification', 'LEDForSequenceClassification',
#  'DistilBertForSequenceClassification',
#  'AlbertForSequenceClassification',
#  'CamembertForSequenceClassification',
#  'XLMRobertaForSequenceClassification',
#  'MBartForSequenceClassification', 'BartForSequenceClassification',
#  'LongformerForSequenceClassification',
#  'RobertaForSequenceClassification',
#  'SqueezeBertForSequenceClassification',
#  'LayoutLMForSequenceClassification', 'BertForSequenceClassification',
#  'XLNetForSequenceClassification',
#  'MegatronBertForSequenceClassification',
#  'MobileBertForSequenceClassification',
#  'FlaubertForSequenceClassification', 'XLMForSequenceClassification',
#  'ElectraForSequenceClassification',
#  'FunnelForSequenceClassification',
#  'DebertaForSequenceClassification',
#  'DebertaV2ForSequenceClassification',
#  'GPT2ForSequenceClassification', 'GPTNeoForSequenceClassification',
#  'OpenAIGPTForSequenceClassification',
#  'ReformerForSequenceClassification', 'CTRLForSequenceClassification',
#  'TransfoXLForSequenceClassification',
#  'MPNetForSequenceClassification', 'TapasForSequenceClassification',
#  'IBertForSequenceClassification', 'XLMRobertaModelWithHeads',
#  'RobertaModelWithHeads', 'BertModelWithHeads',
#  'DistilBertModelWithHeads', 'BartModelWithHeads',
#  'MBartModelWithHeads', 'GPT2ModelWithHeads', 'T5ModelWithHeads'].

# sorted by length of model (to prefer longer matches over shorted ones)
base_models = ["bert-large-uncased-whole-word-masking-finetuned-squad",
               "bert-large-cased-whole-word-masking-finetuned-squad",
               "distilbert-base-uncased-finetuned-sst-2-english",
               "xlm-roberta-large-finetuned-conll03-english",
               "xlm-roberta-large-finetuned-conll02-dutch",
               "distilbert-base-uncased-distilled-squad",
               "bert-large-uncased-whole-word-masking",
               "bert-large-cased-whole-word-masking",
               "bert-base-multilingual-uncased",
               "bert-base-cased-finetuned-mrpc",
               "roberta-large-openai-detector",
               "distilbert-base-german-cased",
               "bert-base-multilingual-cased",
               "xlm-mlm-tlm-xnli15-1024",
               "distilbert-base-uncased",
               "distilbert-base-cased",
               "roberta-large-mnli",
               "distilroberta-base",
               "bert-large-uncased",
               "xlm-roberta-large",
               "xlm-mlm-enfr-1024",
               "xlm-mlm-ende-1024",
               "xlm-clm-enfr-1024",
               "xlm-clm-ende-1024",
               "bert-base-uncased",
               "bert-base-chinese",
               "albert-xxlarge-v2",
               "albert-xxlarge-v1",
               "xlnet-base-cased",
               "xlm-roberta-base",
               "transfo-xl-wt103",
               "bert-large-cased",
               "albert-xlarge-v2",
               "albert-xlarge-v1",
               "xlm-mlm-en-2048",
               "bert-base-cased",
               "albert-large-v2",
               "albert-large-v1",
               "camembert-base",
               "albert-base-v2",
               "albert-base-v1",
               "roberta-large",
               "roberta-base",
               "gpt2-medium",
               "gpt2-large",
               "distilgpt2",
               "t5-small",
               "t5-large",
               "t5-base",
               "gpt2",
               "ctrl",
               'bert-base-uncased',
               'roberta-base',
               'distilbert-base-uncased',
               'distilbert-base-cased',
               'roberta-base',
               'albert-base-v2',
               'xlnet-base-cased',
               'facebook/bart-large',
               'xlnet-large-cased',
               'bert-base-uncased' ]

# horrible hack: we will attempt to infer the base model from the model name
def my_tokenizer(model_key):
    print('my_tokenizer: ' + str(model_key), file=sys.stderr)
    try:
        return AutoTokenizer.from_pretrained(model_key)
    except:
        for base_model_key in base_models:
            if base_model_key in model_key:
                return AutoTokenizer.from_pretrained(base_model_key)
        config = get_config(model_key)
        if not config is None and "_name_or_path" in config:
            return AutoTokenizer.from_pretrained(config["_name_or_path"])
        else:
            print('Warning, cannot find tokenizer for: ' + str(model_key), file=sys.stderr)
            return None


# Here are some of the more important tasks to add
# egrep error `cat $pwd/errors.txt` | awk 'match($0,/task not supported/) {print substr($0, RSTART+RLENGTH+2)}'  | cut -f1 -d, | sort | uniq -c | sort -nr
     # 10 text-to-speech')
     # 10 audio-to-audio')
     #  9 text-to-image')
     #  5 structured-data-classification')
     #  4 sentence-similarity')
     #  2 voice-activity-detection')
     #  1 zero-shot-image-classification')
     #  1 speech-segmentation')
     #  1 protein-folding')



import transformers

# old_my_task_to_auto_class = { 
#     'ASR'                            : transformers.AutoModelForCTC,
#     'MT'                             : transformers.AutoModelWithLMHead,
#     'QA'                             : transformers.AutoModelForQuestionAnswering,
#     'audio-classification'           : transformers.AutoModelForAudioClassification,
#     'automatic-speech-recognition'   : transformers.AutoModelForCTC,
#     'classify'                       : transformers.AutoModelForSequenceClassification,
#     'classify_spans'                 : transformers.AutoModelForQuestionAnswering,
#     'classify_tokens'                : transformers.AutoModelForTokenClassification,
#     'conversational'                 : transformers.AutoModelForCausalLM,
#     'default'                        : transformers.AutoModel,
#     'feature-extraction'             : transformers.AutoFeatureExtractor,
#     'fill-mask'                      : transformers.AutoModelForMaskedLM,
#     'image-classification'           : transformers.AutoModelForImageClassification,
#     'image-segmentation'             : transformers.AutoModelForImageSegmentation,
#     'image-to-text'                  : transformers.AutoModelForVision2Seq,
#     'ner'                            : transformers.AutoModelForTokenClassification,
#     'object-detection'               : transformers.AutoModelForObjectDetection,
#     'question-answering'             : transformers.AutoModelForQuestionAnswering,
#     'sentiment-analysis'             : transformers.AutoModelForSequenceClassification,
#     'speech-segmentation'            : transformers.AutoModelForCTC,
#     'summarization'                  : transformers.AutoModelForSeq2SeqLM,
#     'table-question-answering'       : transformers.AutoModelForTableQuestionAnswering,
#     'text-classification'            : transformers.AutoModelForSequenceClassification,
#     'text-generation'                : transformers.AutoModelForSequenceClassification,
#     'text-to-image'                  : transformers.AutoModelForSeq2SeqLM,
#     'text2text-generation'           : transformers.AutoModelForSeq2SeqLM,
#     'token-classification'           : transformers.AutoModelForTokenClassification,
#     'translation'                    : transformers.AutoModelWithLMHead,
#     'translation_xx_to_yy'           : transformers.AutoModelForSeq2SeqLM,
#     'zero-shot-classification'       : transformers.AutoModelForSequenceClassification, 
#     None                             : transformers.AutoModelForSequenceClassification,
# }

# This convoluted approach makes it possible to use older versions of
# transformers that do not support all of these auto models

my_task_to_auto_class = { 
    'ASR'                            : 'AutoModelForCTC',
    'MT'                             : 'AutoModelWithLMHead',
    'QA'                             : 'AutoModelForQuestionAnswering',
    'audio-classification'           : 'AutoModelForAudioClassification',
    'automatic-speech-recognition'   : 'AutoModelForCTC',
    'classify'                       : 'AutoModelForSequenceClassification',
    'classify_spans'                 : 'AutoModelForQuestionAnswering',
    'classify_tokens'                : 'AutoModelForTokenClassification',
    'conversational'                 : 'AutoModelForCausalLM',
    'default'                        : 'AutoModel',
    'feature-extraction'             : 'AutoFeatureExtractor',
    'fill-mask'                      : 'AutoModelForMaskedLM',
    'image-classification'           : 'AutoModelForImageClassification',
    'image-segmentation'             : 'AutoModelForImageSegmentation',
    'image-to-text'                  : 'AutoModelForVision2Seq',
    'ner'                            : 'AutoModelForTokenClassification',
    'object-detection'               : 'AutoModelForObjectDetection',
    'question-answering'             : 'AutoModelForQuestionAnswering',
    'sentiment-analysis'             : 'AutoModelForSequenceClassification',
    'speech-segmentation'            : 'AutoModelForCTC',
    'summarization'                  : 'AutoModelForSeq2SeqLM',
    'table-question-answering'       : 'AutoModelForTableQuestionAnswering',
    'text-classification'            : 'AutoModelForSequenceClassification',
    'text-generation'                : 'AutoModelForSequenceClassification',
    'text-to-image'                  : 'AutoModelForSeq2SeqLM',
    'text2text-generation'           : 'AutoModelForSeq2SeqLM',
    'token-classification'           : 'AutoModelForTokenClassification',
    'translation'                    : 'AutoModelWithLMHead',
    'translation_xx_to_yy'           : 'AutoModelForSeq2SeqLM',
    'zero-shot-classification'       : 'AutoModelForSequenceClassification', 
    None                             : 'AutoModelForSequenceClassification',
}

# for k in my_task_to_auto_class:
#     v = my_task_to_auto_class[k]
#     if hasattr(transformers, v):
#         my_task_to_auto_class[k] = getattr(transformers,v)

# infer the appropriate automodel from args (and especially the task)
def auto_model_for_X(args):
    from gft.gft_internals.my_task import infer_task
    task = infer_task(args)

    print('task: ' + str(task), file=sys.stderr)
    # task_provider,task = parse_task_specification(args)

    if task in my_task_to_auto_class:
        v = my_task_to_auto_class[task]
        if hasattr(transformers, v):
            return getattr(transformers, v)

    assert False, 'auto_model_for_X, task not supported: ' + task


def old_auto_model_for_X(args):
    from gft.gft_internals.my_task import infer_task
    task = infer_task(args)
    # print('task: ' + str(task), file=sys.stderr)

    # task_provider,task = parse_task_specification(args)

    if task is None:
        from transformers import AutoModelForSequenceClassification
        return AutoModelForSequenceClassification

    if task == 'text-to-image':
        from transformers import AutoModelForSeq2SeqLM
        return AutoModelForSeq2SeqLM

    if task == 'speech-segmentation':
        from transformers import AutoModelForCTC
        return AutoModelForCTC

    if task == 'default':
        from transformers import AutoModel
        return AutoModel

    if task == "audio-classification":
        from transformers import AutoModelForAudioClassification
        return AutoModelForAudioClassification

    if task == "ASR" or task == "automatic-speech-recognition":
        # from transformers import AutoModelForSpeechSeq2Seq
        # return AutoModelForSpeechSeq2Seq
        from transformers import AutoModelForCTC
        return AutoModelForCTC
    
    if task == "conversational":
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM
        # assert False, 'auto_model_for_X, task not supported: ' + task

    if task == "feature-extraction":
        from transformers import AutoFeatureExtractor
        return AutoFeatureExtractor

    if task == "fill-mask":
        from transformers import AutoModelForMaskedLM
        return AutoModelForMaskedLM
        # assert False, 'auto_model_for_X, task not supported: ' + task

    if task == "image-classification":
        from transformers import AutoModelForImageClassification
        return AutoModelForImageClassification

    if task == "QA" or task == "question-answering":
        from transformers import AutoModelForQuestionAnswering
        return AutoModelForQuestionAnswering
    
    if task == "table-question-answering":
        from transformers import AutoModelForTableQuestionAnswering
        return AutoModelForTableQuestionAnswering
        # assert False, 'auto_model_for_X, task not supported: ' + task

    if task == "image-segmentation":
        from transformers import AutoModelForImageSegmentation
        return AutoModelForImageSegmentation

    if task == "object-detection":
        from transformers import AutoModelForObjectDetection
        return AutoModelForObjectDetection

    if task == "text2text-generation":
        from transformers import AutoModelForSeq2SeqLM
        return AutoModelForSeq2SeqLM
        # assert False, 'auto_model_for_X, task not supported: ' + task

    if task == "text-classification" or task == "sentiment-analysis":
        from transformers import AutoModelForSequenceClassification
        return AutoModelForSequenceClassification

    if task == "text-generation":
        from transformers import AutoModelForSequenceClassification
        return AutoModelForSequenceClassification
        # assert False, 'auto_model_for_X, task not supported: ' + task

    if task == "token-classification" or task == "ner":
        from transformers import AutoModelForTokenClassification
        return AutoModelForTokenClassification

    if task == "MT" or task == "translation":
        from transformers import AutoModelWithLMHead
        return AutoModelWithLMHead
        # from transformers import AutoModelForSeq2SeqLM
        # return AutoModelForSeq2SeqLM

    # not sure this is the right automodel for summarization
    if task == "translation_xx_to_yy" or task == "summarization":
        from transformers import AutoModelForSeq2SeqLM
        return AutoModelForSeq2SeqLM

    # not sure this is the right automodel for image-to-text
    if task == "image-to-text":
        from transformers import AutoModelForVision2Seq
        return AutoModelForVision2Seq

    if task == "zero-shot-classification":
        from transformers import AutoModelForSequenceClassification
        return AutoModelForSequenceClassification

    assert False, 'auto_model_for_X, task not supported: ' + task

def my_load_model_tokenizer_and_extractor(args, keyword='model'):
    base_model_provider, base_model_key, model_provider, model_key = parse_base_model_specification(args)

    # model_provider,model_key = parse_model_specification(args, keyword=keyword)
    # base_model_provider,base_model_key = parse_model_specification(args, keyword='base_model')
    # print('model_provider: ' + str(model_provider), file=sys.stderr)

    if model_key is None: return None,None,None
        
    a = my_get_adapter_info(model_key)
    print('a: ' + str(a), file=sys.stderr)
    if a is None or a.model_name is None:

        print('model_provider: ' + str(model_provider), file=sys.stderr)
        print('model_key: ' + str(model_key), file=sys.stderr)

        base_model = auto_model_for_X(args)
        print('base_model: ' + str(base_model), file=sys.stderr)

        
        model = base_model.from_pretrained(model_key, return_dict=True)

        tokenizer = None
        extractor = None
        try:
            if not base_model_key is None: tokenizer = my_tokenizer(base_model_key)
            else:tokenizer = my_tokenizer(model_key)
        except:
            tokenizer=None
        try:
            extractor = AutoFeatureExtractor.from_pretrained(model_key)
        except:
            extractor=None
        return model,tokenizer,extractor

    # adaptors
    m = a.model_name
    tokenizer=None
    extractor=None
    try:
        tokenizer = AutoTokenizer.from_pretrained(a.model_name)
    except:
        tokenizer=None

    try:
        extractor = AutoFeatureExtractor.from_pretrained(a.model_name)
    except:
        extractor=None

    if 'distilbert' in a.model_name:
        from transformers import DistilBertModelWithHeads
        model = DistilBertModelWithHeads.from_pretrained(a.model_name)
    elif 'bart' in a.model_name:
        from transformers import BartModelWithHeads
        model = BartModelWithHeads.from_pretrained(a.model_name)
    elif 'roberta' in a.model_name:
        from transformers import RobertaModelWithHeads
        model = RobertaModelWithHeads.from_pretrained(a.model_name)
    elif 'bert' in a.model_name:
        from transformers import BertModelWithHeads        
        model = BertModelWithHeads.from_pretrained(a.model_name)
    else:
        assert False, 'my_load_model_tokenizer_and_extractor: failed for model: ' + str(a.model_name)

    print('model_key: ' + str(model_key), file=sys.stderr)
    print('a.source: ' + str(a.source), file=sys.stderr)
    model.set_active_adapters(model.load_adapter(model_key, source=a.source))
    return model,tokenizer,extractor
