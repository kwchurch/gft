
# adapters are really powerful, but unfortunately, they are a bit tricky to use with automodels

# assert False, 'need to finish work on my_auto_model_pd'

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
from gft_internals.gft_util import parse_model_specification, parse_metric_specification, parse_dataset_specification, better, checkpoint_filename, get_arg, set_arg

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
        config = get_config(model_key)
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


def my_load_model_tokenizer_and_extractor_pd(args, keyword):
    # import pdb
    model_provider,model_key = parse_model_specification(args, keyword=keyword)
    if model_key is None: return None,None,None
    print('model_provider: ' + str(model_provider), file=sys.stderr)
    print('model_key: ' + str(model_key), file=sys.stderr)
    # pdb.set_trace()
    base_model = auto_model_for_X_pd(args)
    print('base_model: ' + str(base_model), file=sys.stderr)
    # pdb.set_trace()
    # model = base_model.from_pretrained(model_key, return_dict=True)
    model = base_model.from_pretrained(model_key)
    # pdb.set_trace()
    tokenizer = None
    extractor = None
    try:
        tokenizer = my_tokenizer(model_key)
    except:
        tokenizer=None
    # try:
    #     extractor = AutoFeatureExtractor.from_pretrained(model_key)
    # except:
    #     extractor=None
    return model,tokenizer,extractor

