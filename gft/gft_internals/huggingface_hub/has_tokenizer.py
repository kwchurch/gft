#!/usr/bin/env python

from transformers import AutoTokenizer
import sys

def has_tokenizer(m):
    try: 
        AutoTokenizer.from_pretrained(m)
        return True
    except:
        return False

for line in sys.stdin:
    fields = line.rstrip().split('\t')
    if len(fields) > 0:
        print('\t'.join(map(str, [fields[0], has_tokenizer(fields[0])])))
