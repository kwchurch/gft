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

keys = ['id', 'downloads', 'paperswithcode_id', 'tags']
print('\t'.join(keys))

def dataset2str(d):
    return '\t'.join([geta(d, a) for a in keys])

for d in huggingface_hub.list_datasets():
    print(dataset2str(d))
    
