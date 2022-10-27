#!/usr/bin/env python

import argparse, datasets

parser = argparse.ArgumentParser(description="Extract text from a dataset")
parser.add_argument("--text_field", type=str, help="name of the text file in the dataset", default='test')
parser.add_argument("--split", type=str, help="name of the split in the dataset", required=True)
parser.add_argument("--config", type=str, help="name of the config in the dataset", default=None)
parser.add_argument("--data", type=str,
        help="A dataset from HuggingFace (https://huggingface.co/datasets) or a custom dataset (--csv_dataset)",
        required=True)
args = parser.parse_args()
wrapped_args = { "wrapper" : args }

if args.config is None:
    ds = datasets.load_dataset(wrapped_args, split=args.split)
else:
    ds = datasets.load_dataset(wrapped_args, args.config, split=args.split)

for record in ds:
    print(record[args.text_field])
    
