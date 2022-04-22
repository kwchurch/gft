#!/usr/bin/env python

# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# from https://github.com/huggingface/accelerate/blob/main/examples/nlp_example.py
# modified by author of paper under review

# https://discuss.huggingface.co/t/fine-tune-for-multiclass-or-multilabel-multiclass/4035

# import pdb

import sys

from accelerate import Accelerator

# sys.path.append(os.environ.get('gft') + '/gft_internals')

# from .gft_internals.gft_util import get_arg

# from .gft_internals import my_datasets
# from .gft_internals import parse_args, parse_eqn

from gft.gft_internals.gft_util import get_arg

from gft.gft_internals import my_datasets
from gft.gft_internals import parse_args, parse_eqn


def training_function(args):
    accelerator = Accelerator(fp16=get_arg(args, 'fp16'), cpu=get_arg(args, 'cpu', default=False))
    eqn = parse_eqn.parse_eqn(get_arg(args, 'eqn'))

    # output_dir = get_arg(args, 'output_dir')
    # if not output_dir is None:
    #     print('output_dir: ' + str(output_dir), file=sys.stderr)
    #     with open(os.path.join(output_dir, 'eqn.json'), 'w') as fd:
    #         json.dump(eqn,fd)
    #     with open(os.path.join(output_dir, 'args.txt'), 'w') as fd:
    #         print(args, file=fd)

    # print('eqn: ' + str(eqn))
    eqn_type = eqn['eqn_type']

    datasets = my_datasets.my_load_dataset(args)

    # pdb.set_trace()
    # parse_eqn.verify_dataset_columns(datasets, eqn)

    # pdb.set_trace()

    # Currently, only a few values for eqn_type are supported, but we plan to add more soonish
    # See parse_eqn.eqn_types_list for list of types that we would like to support

    # classify should cover classify_classic and classify_regress (but not all cases are working yet)
    if eqn_type == parse_eqn.eqn_types['classify']:
        from gft.gft_internals import eval_for_classify
        # from .gft_internals import eval_for_classify
        eval_for_classify.my_eval(args, eqn, accelerator, datasets)

    elif eqn_type == parse_eqn.eqn_types['classify_classic']:
        from gft.gft_internals import eval_for_classify_classic
        # from .gft_internals import eval_for_classify_classic
        eval_for_classify_classic.my_eval(args, eqn, accelerator, datasets)

    elif eqn_type == parse_eqn.eqn_types['regress']:
        # from .gft_internals  import eval_for_regress
        from gft.gft_internals  import eval_for_regress
        eval_for_regress.my_eval(args, eqn, accelerator, datasets)

        # if len(eqn['y_field_names']) == 1:
        #     fit_for_classify.fit(args, eqn, accelerator, datasets, is_regression=True)
        # else:
        #     # we should generalize fit_for_classify so it works with more than one y_field_name
        #     fit_for_regress.fit(args, eqn, accelerator, datasets)

    elif eqn_type == parse_eqn.eqn_types['classify_tokens']:
        from gft.gft_internals import eval_for_classify_tokens
        # from .gft_internals import eval_for_classify_tokens
        eval_for_classify_tokens.my_eval(args, eqn, accelerator, datasets)

    elif eqn_type == parse_eqn.eqn_types['classify_spans']:
        from gft.gft_internals import eval_for_classify_spans
        # from .gft_internals import eval_for_classify_spans
        eval_for_classify_spans.my_eval(args, eqn, accelerator, datasets)

    elif eqn_type == parse_eqn.eqn_types['ctc']:
        from gft.gft_internals import eval_for_ctc
        # from .gft_internals import eval_for_ctc
        eval_for_ctc.my_eval(args, eqn, accelerator, datasets)

    else:
        assert False, 'model for eqn_type: %s is not implemented (yet)' % str(
            parse_eqn.eqn_types_list[eqn_type])  # stub to flesh out later

def main():

    # import pdb
    # pdb.set_trace()

    # horrible hack to get around
    # requirement that --output_dir be specified
    sys.argv.append('--output_dir')
    sys.argv.append(None)

    # attempt to support all args in gft
    args = parse_args.gft_parse_args()

    # pdb.set_trace()

    # outdir = get_arg(args, 'output_dir')
    # if outdir: os.makedirs(outdir, exist_ok=True)

    # print('args: ' + str([i for i in args]))
    # training_function(args)

    if get_arg(args, 'do_not_catch_errors'): 
        training_function(args)
    else:
        try: 
            training_function(args)
        except:
            model = str(get_arg(args, 'model'))
            data = str(get_arg(args, 'data'))
            eqn = str(get_arg(args, 'eqn'))
            task = str(get_arg(args, 'task', default=None))
            splits = str(get_arg(args, 'splits', default=None))
            split = str(get_arg(args, 'split', default=None))
            print('\t'.join(['***ERROR***', 'model: ' + str(model), 'data: ' + str(data), 'eqn: ' + str(eqn),
                         'splits: ' + str(split), 'split: ' + str(split), 'task: ' + str(task) , 'error: ' + str(sys.exc_info())]))

if __name__ == "__main__":
    main()
