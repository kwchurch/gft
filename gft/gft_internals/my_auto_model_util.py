import sys
import gft
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

