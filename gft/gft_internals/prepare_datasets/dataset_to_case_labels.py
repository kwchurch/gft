#!/usr/bin/env python

from tqdm import tqdm

# import pickle

import pandas as pd

import sys,re,argparse,datasets,pdb,os
# sys.path.append('.')
# sys.path.append('..')

# gft = os.environ.get('gft')
# assert not gft is None, 'please set the environment variable, gft, appropriately, and export it'
# sys.path.append(gft)

import gft
from gft.gft_internals import my_datasets

parser = argparse.ArgumentParser(description="Extract text from a dataset")
parser.add_argument("--text_field", type=str, help="name of the text file in the dataset", required=True)
parser.add_argument("--splits", type=str, help="3 comma separate values for train,val,test (if dataset does not have a val set, use __select_from_train__ for val)", default="train,val,test")
parser.add_argument("--config", type=str, help="name of the config in the dataset", default=None)
parser.add_argument("--output", type=str, help="name of output file", required=True)
parser.add_argument("--data", type=str,
        help="A dataset from HuggingFace (https://huggingface.co/datasets) or a custom dataset (--csv_dataset)",
        required=True)

parser.add_argument("--data_dir", type=str,
                    help="optional argument to HuggingFace datasets.load_dataset (usually not needed)",
                    default=None)

parser.add_argument("--debug", action="store_true", help="Activate debug mode and run training only with a subset of data.")
args = parser.parse_args()
wrapped_args = { "wrapper" : args }

ds = my_datasets.my_load_dataset(wrapped_args)

def my_str(x, T):
    s = str(x)
    if len(s) > T: return s[0:T] + '+'
    else: return s

vocab = ["x", "xo", "Xx", "xox", "X", "Xxo", "ox", "oXx", "o", "Xxox", "oXxo", "oxo", "Xox", "Xo", "oX", "xoxo", "xoXx", "xoxox", "oXxox", "XxoXx"]
ivocab={e: i for i,e in enumerate(vocab)}

def letter_label(letter):
    if letter.isupper(): return 'X'
    if letter.islower(): return 'x'
    return 'o'    

def word_label(w):
    label = ''.join([letter_label(l) for l in w])
    label = re.sub('x+', 'x', label)
    label = re.sub('X+', 'X', label)
    label = re.sub('o+', 'o', label)
    if label in ivocab: return label
    else: return 'misc'

for split in ['train', 'val', 'test']:
    print('split: ' + split, file=sys.stderr)
    tokens_col = []
    case_labels_col = []
    for r in tqdm(ds[split]):
        tokens = r[args.text_field].split()
        # print('tokens: ' + my_str(tokens,150), file=sys.stderr)
        labels = [ word_label(w) for w in tokens ]
        # print('tokens: ' + my_str(labels,150), file=sys.stderr)

        tokens_col.append(' '.join(tokens))
        case_labels_col.append(' '.join(labels))

    d = pd.DataFrame.from_dict({'tokens' : tokens_col, 'case_labels' : case_labels_col})
    d.to_csv(args.output + '.' + split, index=False)
        
    
# with open(args.output, 'wb') as fd:
#     pickle.dump(ds, fd, protocol=pickle.HIGHEST_PROTOCOL)
    

