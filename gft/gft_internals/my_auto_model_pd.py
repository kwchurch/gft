
# adapters are really powerful, but unfortunately, they are a bit tricky to use with automodels

# assert False, 'need to finish work on my_auto_model_pd'
import gft
import sys,os

# AutoModelForCausalLM
# AutoModelForConditionalGeneration
# AutoModelForMaskedLM
# AutoModelForMultipleChoice
# AutoModelForPretraining
# AutoModelForQuestionAnswering
# AutoModelForSequenceClassification
# AutoModelForTokenClassification

import paddle
from paddlenlp.transformers import (
    # AutoModelForSequenceClassification as AutoModelForSequenceClassification_pd,
    AutoTokenizer as AutoTokenizer_pd,
)

# import transformers.adapters.utils
from gft.gft_internals.gft_util import parse_base_model_specification, parse_metric_specification, parse_dataset_specification, better, checkpoint_filename, get_arg, set_arg

def get_config_pd(fn):
    p = os.path.join(fn, 'config.json')
    try:
        with open(p, 'r') as fd:
            import json
            return json.loads(fd.read())
    except:
        print('Warning, cannot find file: ' + str(p), file=sys.stderr)
        return None

def my_tokenizer_pd(model_key):
    try:
        return AutoTokenizer_pd.from_pretrained(model_key)
    except:
        config = get_config_pd(model_key)
        if hasattr(config, "_name_or_path"):
            return AutoTokenizer_pd.from_pretrained(config["_name_or_path"])
        else:
            print('Warning, cannot find tokenizer for: ' + str(model_key), file=sys.stderr)
            return None

def auto_model_for_X_pd(args):
    task = get_arg(args, 'task', default=None)
    if task is None:
        # from paddlenlp.transformers import AutoModelForSequenceClassification
        # return AutoModelForSequenceClassification
        from paddlenlp.transformers import AutoModel
        return AutoModel
    if task == "audio-classification":
        from paddlenlp.transformers import AutoModelForAudioClassification
        return AutoModelForAudioClassification

    if task == "ASR" or task == "automatic-speech-recognition":
        assert False, 'auto_model_for_X_pd, task not supported: ' + task
        # from paddlenlp.transformers import AutoModelForSpeechSeq2Seq
        # return AutoModelForSpeechSeq2Seq
    
    if task == "conversational":
        assert False, 'auto_model_for_X_pd, task not supported: ' + task

    if task == "feature-extraction":
        assert False, 'auto_model_for_X_pd, task not supported: ' + task

    if task == "fill-mask":
        assert False, 'auto_model_for_X_pd, task not supported: ' + task

    if task == "image-classification":
        assert False, 'auto_model_for_X_pd, task not supported: ' + task
        # from paddlenlp.transformers import AutoModelForImageClassification
        # return AutoModelForImageClassification

    if task == "QA" or task == "question-answering":
        from paddlenlp.transformers import AutoModelForQuestionAnswering
        return AutoModelForQuestionAnswering
    
    if task == "table-question-answering":
        assert False, 'auto_model_for_X_pd, task not supported: ' + task

    if task == "text2text-generation":
        assert False, 'auto_model_for_X_pd, task not supported: ' + task

    if task == "text-classification" or task == "sentiment-analysis":
        from paddlenlp.transformers import AutoModelForSequenceClassification
        return AutoModelForSequenceClassification

    if task == "text-generation":
        assert False, 'auto_model_for_X_pd, task not supported: ' + task

    if task == "token-classification" or task == "ner":
        from paddlenlp.transformers import AutoModelForTokenClassification
        return AutoModelForTokenClassification

    if task == "MT" or task == "translation":
        from paddlenlp.transformers import AutoModelWithLMHead
        return AutoModelWithLMHead
        # from transformers import AutoModelForSeq2SeqLM
        # return AutoModelForSeq2SeqLM

    if task == "translation_xx_to_yy":
        assert False, 'auto_model_for_X_pd, task not supported: ' + task
        # from paddlenlp.transformers import AutoModelForSeq2SeqLM
        # return AutoModelForSeq2SeqLM

    if task == "summarization":
        assert False, 'auto_model_for_X_pd, task not supported: ' + task

    if task == "zero-shot-classification":
        assert False, 'auto_model_for_X_pd, task not supported: ' + task

    assert False, 'auto_model_for_X_pd, task not supported: ' + task

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
        return AutoTokenizer_pd.from_pretrained(model_key)
    except:
        for base_model_key in base_models:
            if base_model_key in model_key:
                return AutoTokenizer_pd.from_pretrained(base_model_key)
        config = get_config_pd(model_key)
        if not config is None and "_name_or_path" in config:
            return AutoTokenizer_pd.from_pretrained(config["_name_or_path"])
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



def my_load_model_tokenizer_and_extractor_pd(args, keyword):
    base_model_provider, base_model_key, model_provider, model_key = parse_base_model_specification(args)

    if model_key is None: return None,None,None
    print('model_provider: ' + str(model_provider), file=sys.stderr)
    print('model_key: ' + str(model_key), file=sys.stderr)
    base_model = auto_model_for_X_pd(args)
    print('base_model: ' + str(base_model), file=sys.stderr)
    model = base_model.from_pretrained(model_key)
    tokenizer = None
    extractor = None

    if not base_model_key is None: 
        tokenizer = my_tokenizer(base_model_key)
    else:
        tokenizer = my_tokenizer(model_key)


    return model,tokenizer,extractor

