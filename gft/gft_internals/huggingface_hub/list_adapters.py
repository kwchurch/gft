#!/usr/bin/env python

import transformers.adapters.utils
import sys

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

keys = ['adapter_id', 'source', 'model_name', 'task', 'subtask']
print('\t'.join(keys))

def adapter2str(a):
    return '\t'.join(map(str, [geta(a, at) for at in keys]))

for a in transformers.adapters.utils.list_adapters():
    print(adapter2str(a))
