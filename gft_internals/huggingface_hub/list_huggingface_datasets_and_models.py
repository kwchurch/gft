#!/usr/bin/env python

import sys
from huggingface_hub import HfApi, ModelFilter, ModelSearchArguments

model_args = ModelSearchArguments()

for i,d in enumerate(model_args.dataset):
    api = HfApi()
    models = api.list_models(filter=(d))
    print('# i: %d, dataset: %s, found %d models' % (i, d, len(models)))
    sys.stdout.flush()
    for m in models:
        print('\t'.join([str(d), str(m.modelId)]))
