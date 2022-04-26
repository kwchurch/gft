
# based on https://github.com/huggingface/transformers/blob/master/examples/pytorch/text-classification/run_glue_no_trainer.py

# convention: _hf is from HuggingFace, and _pd is from PaddleHub

import gft
import sys

def fit(args, eqn, accelerator, raw_datasets):

    from gft.gft_internals.gft_util import parse_model_specification
    provider,model_key = parse_model_specification(args)
    
    print(__name__ + ': provider = ' + str(provider), file=sys.stderr)

    if provider == 'PaddleHub':
        from gft.gft_internals.fit_for_classify_spans_pd import fit as fit_pd
        return fit_pd(args, eqn, accelerator, raw_datasets)
    else:
        from gft.gft_internals.fit_for_classify_spans_hf import fit as fit_hf
        return fit_hf(args, eqn, accelerator, raw_datasets)
