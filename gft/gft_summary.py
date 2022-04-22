#!/usr/bin/env python

# guess set of labels from a model or a dataset
#
# examples of usage:
# gft_summary --model 'H:AdapterHub/bert-base-uncased-pf-emotion'
# gft_summary --data 'H:emotion'
# gft_summary --data 'H:emotion' --model 'H:__infer__'

import argparse
import numpy as np
import os,sys,json
import torch

# print('loading gft_summary.py', file=sys.stderr)

# sys.path.append(os.environ.get('gft') + '/gft_internals')

from gft.gft_internals.gft_util import parse_dataset_specification,parse_model_specification
# from gft_internals.gft_util import parse_dataset_specification,parse_model_specification

def do_it(args):

    data_provider,data_key = parse_dataset_specification(args)
    model_provider,model_key = parse_model_specification(args)

    print('data_provider: ' + str(data_provider), file=sys.stderr)
    print('model_provider: ' + str(model_provider), file=sys.stderr)

    if model_provider == 'PaddleHub' or data_provider == 'PaddleHub':
        from gft.gft_internals import gft_summary_pd
        # from .gft_internals import gft_summary_pd
        return gft_summary_pd.gft_summary_pd(args)
    else:
        from gft.gft_internals import gft_summary_hf
        # from .gft_internals import gft_summary_hf
        return gft_summary_hf.gft_summary_hf(args)


def main():
    parser = argparse.ArgumentParser(description="Summarize model and/or data; if model is __infer__, produce a list of models based on data or task")
    parser.add_argument("--model", type=str, help="prefix (H/P/C): base model | checkpoint | adapter | __infer__", default=None)
    parser.add_argument("--data", type=str, help="prefix (H/P/C): dataset name", default=None)
    parser.add_argument("--eqn", type=str, help="example: classify: labels ~ sentence1 + sentence2", default=None)
    parser.add_argument("--split", type=str, help="test,val,train,...", default='train')
    parser.add_argument("--splits", type=str, help="train,val,test (3 comma separated values)", default=None)
    parser.add_argument("--task", type=str, help="eqn type (classify, classify_tokens, classify_spans, ctc, etc.) or a pipeline task (text-classification, translation, ASR, etc.)", default=None)
    parser.add_argument("--topn", type=int, help="number of models to return [defaults to 10]", default=10)
    parser.add_argument("--do_not_catch_errors", action="store_true", help="Do not catch errors")
    parser.add_argument("--fast_mode", action="store_true", help="Turn on fast mode (go easy on API calls)")

    args = parser.parse_args()
    wrapped_args = { "wrapper" : args }

    # print(args)

    if args.do_not_catch_errors:
        do_it(wrapped_args)
        return
    try:
        do_it(wrapped_args)
    except:
        print('\n'.join(['***ERROR***', 'model: ' + str(args.model), 'data: ' + str(args.data), 'eqn: ' + str(args.eqn),
                         'splits: ' + str(args.split), 'split: ' + str(args.split), 'task: ' + str(args.task) , 'error: ' + str(sys.exc_info())]))

if __name__ == "__main__":
    main()
