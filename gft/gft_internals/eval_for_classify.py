import gft

# based on https://github.com/huggingface/transformers/blob/master/examples/pytorch/text-classification/run_glue_no_trainer.py

# convention: _hf is from HuggingFace, and _pd is from PaddleHub

# TODO: Adapters
# https://github.com/Adapter-Hub/adapter-transformers/blob/master/examples/adapterfusion/run_fusion_glue.py

import sys

def my_eval(args, eqn, accelerator, raw_datasets, is_regression=False):

    from gft.gft_internals.gft_util import parse_model_specification
    provider,model_key = parse_model_specification(args, keyword='model')
    
    print(__name__ + ': provider = ' + str(provider), file=sys.stderr)

    if provider == 'PaddleHub':
        from gft.gft_internals.eval_for_classify_pd import my_eval as my_eval_pd
        return my_eval_pd(args, eqn, accelerator, raw_datasets, is_regression)
    else:
        from gft.gft_internals.eval_for_classify_hf import my_eval as my_eval_hf
        return my_eval_hf(args, eqn, accelerator, raw_datasets, is_regression)
