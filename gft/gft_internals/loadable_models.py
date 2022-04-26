
# It turns out that many of the models on HuggingFace hub cannot be loaded
import gft
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import sys
from gft.gft_internals import my_auto_model_hf

def loadable(m):
    args = {'extras' : {'model': m }}
    try:
        my_auto_model_hf.my_load_model_tokenizer_and_extractor(args, 'model')
        return True
    except:
        return False
    
for line in sys.stdin:
    if len(line) > 1:
        model = line.rstrip()
        print(model + '\t' + str(loadable(model)))
