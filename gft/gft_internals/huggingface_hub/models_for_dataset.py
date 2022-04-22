#!/usr/bin/env python

import sys
from huggingface_hub import HfApi, ModelFilter, ModelSearchArguments

model_args = ModelSearchArguments()

d = model_args.dataset
if not hasattr(d, sys.argv[1]):
    print('# dataset: %s --> 0 models' % (sys.argv[1]))
else:
    d = getattr(d, sys.argv[1])
    api = HfApi()
    models = api.list_models(filter=(d))
    print('# dataset: %s --> %d models' % (sys.argv[1], len(models)))
    for m in models:
        print('\t'.join([sys.argv[1], str(m.modelId)]))
