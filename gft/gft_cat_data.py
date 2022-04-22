#!/usr/bin/env python

import argparse

import sys,json,os

# sys.path.append(os.environ.get('gft') + '/gft_internals')

# from .gft_internals import my_datasets, parse_eqn
from gft.gft_internals import my_datasets, parse_eqn


def x_values(row, x_field_names):
    return '|'.join([str(row[n]) for n in x_field_names])

def y_values(row, y_field_names):
    return '|'.join([str(row[n] if n in row else None) for n in y_field_names])

def my_str(obj, max_len):
    res = str(obj)
    if not max_len is None and len(res) > max_len:
        res = res[0:max_len] + '...'
    if '\n' in res: res.replace('\n', '\\n')
    if '\r' in res: res.replace('\r', '\\r')
    if '\t' in res: res.replace('\t', '\\t')
    return res

def main():
    parser = argparse.ArgumentParser(description="Load test set and output to stdout")

    parser.add_argument("--delimiter", type=str, help="defaults to tab", default='\t')

    parser.add_argument("--split", type=str, help="defaults to test", default='test')

    parser.add_argument("-d", "--data", type=str,
                        help="A dataset from HuggingFace (https://huggingface.co/datasets), PaddlePaddle or a custom dataset (--csv_dataset) -- may be prefaced by H:, P: or C: to disambiguate",
                        required=True)

    parser.add_argument("--data_dir", type=str,
        help="optional argument to HuggingFace datasets.load_dataset (usually not needed)",
        default=None)

    parser.add_argument("-e", "--eqn", type=str,
        help="classify: y ~ text1 + text2, \
                        where the variables are the names of fields in csv files (spaces are not allowed in field names).\
                        Here are some more examples:\
                        regress: y ~ text1 + text2 \
                        regress: Valence + Arousal + Dominance ~ word",
        default=None)

    # If the dataset does not provide train, val and test splits, or if they are called something else,
    # this argument can be used to specify the names of the 3 splits
    # There is a special keyword,  __select_from_train__, when there is no val set
    parser.add_argument("-s", "--splits", type=str,
        help="3 comma separate values for train,val,test (if dataset does not have a val set, use __select_from_train__ for val)",
        default=None)

    parser.add_argument("--max_len", type=int, help="truncate fields that exceed this value", default=None)

    # parser.add_argument("--do_not_catch_errors", action="store_true", help="Do not catch errors")

    args = parser.parse_args()
    wrapped_args = { "wrapper" : args }

    datasets = my_datasets.my_load_dataset(wrapped_args)
    testset = datasets[args.split]

    if args.eqn is None:
        for row in testset:
            print(args.delimiter.join(['%s|%s' % (str(key), my_str(row[key], args.max_len)) for key in row]))
    else:
        eqn = parse_eqn.parse_eqn(args.eqn)
        print('eqn: ' + str(eqn), file=sys.stderr)
        eqn_type = eqn['eqn_type']


        x_field_names = eqn['x_field_names']
        y_field_names = eqn['y_field_names']
        
        for row in testset:
            print(x_values(row, x_field_names) + args.delimiter + y_values(row, y_field_names))
        
if __name__ == "__main__":
    main()
