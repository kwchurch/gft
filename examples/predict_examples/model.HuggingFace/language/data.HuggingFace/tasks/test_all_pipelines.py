#!/usr/bin/env python

from transformers import pipeline
import sys

tasks = ['audio-classification', 
         'automatic-speech-recognition',
         'conversational', 
         'feature-extraction', 
         'fill-mask',
         'image-classification',
         'image-segmentation', 
         'ner',
         'object-detection', 
         'question-answering', 
         'sentiment-analysis',
         'summarization',
         'table-question-answering', 
         'text-classification',
         'text-generation', 
         'text2text-generation',
         'token-classification',
         'translation',
         'zero-shot-classification',
         'zero-shot-image-classification',
         'translation_XX_to_YY']

model_for_task = {"audio-classification" : "superb/wav2vec2-base-superb-ks",
                  'automatic-speech-recognition' : "facebook/wav2vec2-base-960h",
                  "conversational": "microsoft/DialoGPT-small",
                  "feature-extraction": "openai/clip-vit-base-patch32",
                  "fill-mask": "bert-base-uncased",
                  'image-classification' : "google/vit-base-patch16-224",
                  "image-segmentation": "hf-internal-testing/tiny-random-beit-pipeline",
                  "ner": 'dbmdz/bert-large-cased-finetuned-conll03-english',
                  "object-detection": "facebook/detr-resnet-50",
                  'question-answering' : "deepset/roberta-base-squad2",
                  "sentiment-analysis": 'distilbert-base-uncased-finetuned-sst-2-english',
                  "summarization": "google/pegasus-xsum",
                  "table-question-answering": "google/tapas-base-finetuned-wtq",
                  "text-classification": "cross-encoder/ms-marco-MiniLM-L-12-v2",
                  "text-generation": "distilgpt2",
                  "text2text-generation": "facebook/m2m100_418M",
                  "token-classification": "xlm-roberta-large-finetuned-conll03-english",
                  "translation": "Helsinki-NLP/opus-mt-zh-en",
                  "zero-shot-classification": "facebook/bart-large-mnli",
                  "zero-shot-image-classification": "hf-internal-testing/tiny-random-clip-zero-shot-image-classification",
                  "translation_XX_to_YY": "Helsinki-NLP/opus-mt-zh-en",

                  # "sentence-similarity": "sentence-transformers/paraphrase-MiniLM-L6-v2",
                  # "protein-folding": "google-deepmind/alphafold-v2",
                  # "audio-to-audio" : "JorisCos/DPRNNTasNet-ks2_Libri1Mix_enhsingle_16k",
                  # "speech-segmentation": "osanseviero/hubert-sd",
                  # "structured-data-classification": "scikit-learn-examples/example",
                  # "text2text-generation": "facebook/m2m100_418M",
                  # "text-to-image": "flax-community/dalle-mini",
                  # "text-to-speech": "facebook/fastspeech2-en-ljspeech",
                  # "token-classification": "xlm-roberta-large-finetuned-conll03-english",
                  # "Translation": "Helsinki-NLP/opus-mt-zh-en",
                  # "voice-activity-detection": "pyannote/segmentation",
                  # "image-to-text": "kha-white/manga-ocr-base",
}


def do_it(task, args={}):
    try:
        pipe = pipeline(task, **args)
        return pipe.model.name_or_path
    except:
        return '***ERROR***'

for task in model_for_task:
    model = model_for_task[task]
    
    print('\t'.join(map(str, ['task: ' + task, 'pipe.model: ' + str(do_it(task))])))
    print('\t'.join(map(str, ['task: ' + task, 
                              'model: ' + str(model),
                              'pipe.model: ' + str(do_it(task, {'model' : model }))])))
    sys.stdout.flush()
    
