#!/usr/bin/env python

import sys
from paddlenlp.transformers import *

def loadable(autoclass, model, args={}):
    try:
        autoclass.from_pretrained(model, **args)
        return True
    except:
        return False

models = ["ernie-1.0",
          "ernie-2.0-en-finetuned-squad",
          "ernie-2.0-en",
          "ernie-2.0-large-en",
          "ernie-doc-base-en",
          "ernie-doc-base-zh",
          "ernie-gen-base-en",
          "ernie-gen-large-en-430g",
          "ernie-gen-large-en",
          "ernie-gram-zh",
          "ernie-tiny",
          "skep_ernie_1.0_large_ch",
          "skep_ernie_2.0_large_en", ]

autoclass=AutoModel
print('\nAutoModel\n')

for model in models:
    print('\t'.join(map(str, ['AutoModel', model, loadable(autoclass, model)])))
    sys.stdout.flush()

autoclass=AutoModelForSequenceClassification
print('\nAutoModelForSequenceClassification\n')

for model in models:
    print('\t'.join(map(str, ['AutoModelForSequenceClassification', model, loadable(autoclass, model, {'num_labels':2})])))
    sys.stdout.flush()

    

    
    
    
