import gft
# based on https://github.com/huggingface/transformers/blob/master/examples/pytorch/text-classification/run_glue_no_trainer.py

# convention: _hf is from HuggingFace, and _pd is from PaddleHub

def my_eval(args, eqn, accelerator, raw_datasets):
    from gft.gft_internals.gft_util import parse_model_specification
    provider,model_key = parse_model_specification(args)
    
    # print('provider: ' + str(provider))

    if provider == 'PaddleHub':
        from gft.gft_internals.eval_for_regress_pd import my_eval
        return my_eval(args, eqn, accelerator, raw_datasets)
    else:
        from gft.gft_internals.eval_for_regress_hf import my_eval
        return my_eval(args, eqn, accelerator, raw_datasets)
