#!/usr/bin/env python

import sys,huggingface_hub

def strify(x):
    if isinstance(x, list):
        return '|'.join(x)
    else:
        return str(x)

def geta(x, attr):
    if hasattr(x, attr):
        return strify(getattr(x, attr))
    else:
        return ''

keys = ['modelId', 'pipeline_tag', 'tags']
print('\t'.join(keys))

def model2str(m):
    return '\t'.join([geta(m, a) for a in keys])


for m in huggingface_hub.list_models():
    print(model2str(m))
    
