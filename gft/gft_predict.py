#!/usr/bin/env python

import argparse
import sys

# sys.path.append(os.environ.get('gft') + '/gft_internals')

from gft.gft_internals.gft_util import parse_task_specification,parse_model_specification
# from .gft_internals.gft_util import parse_task_specification,parse_model_specification

def main():
    parser = argparse.ArgumentParser(description="Simple example of predict script.")
    parser.add_argument("--model", type=str, help="prefix (H/P/C): base model | checkpoint | adapter", default=None)
    parser.add_argument("--base_model", type=str, help="prefix (H/P/C): base model | checkpoint | adapter", default=None)
    parser.add_argument("--data", type=str, help="prefix (H/P/C): dataset name", default=None)
    parser.add_argument("--data_dir", type=str, help="optional argument to HuggingFace datasets.load_dataset (usually not needed)", default=None)
    parser.add_argument("--eqn", type=str, help="example: classify: labels ~ sentence1 + sentence2", default=None)
    parser.add_argument("--split", type=str, help="train,val,test, etc.", default=None)
    parser.add_argument("--task", type=str, help="see https://huggingface.co/docs/transformers/v4.16.2/en/main_classes/pipelines#transformers.pipeline.task", default=None)
    # parser.add_argument("--adapter", type=str, help="adpater (https://github.com/Adapter-Hub/adapter-transformers)", default=None)
    parser.add_argument("--labels", type=str, help="file containing labels", default=None)
    parser.add_argument("--delimiter", type=str, help="defaults to tab", default='\t')
    parser.add_argument("--do_not_catch_errors", action='store_true')
    parser.add_argument("--evaluate", action='store_true')

    # we can infer this when the time comes
    # parser.add_argument("--is_split_into_words", action='store_true')

    parser.add_argument("--max_length", type=int, help="used by text-generation pipeline", default=30)
    parser.add_argument("--num_return_sequences", type=int, help="used by text-generation pipeline", default=3)

    # parser.add_argument("--metric", type=str, help="examples: H:accuracy, C:<filename to a custom metric>; <provider string>:key, where provider string is HuggingFace (H), PaddlePaddle (P) or Custome (C); See https://huggingface.co/docs/datasets/using_metrics.html for HuggingFace metrics", default=None)
    # parser.add_argument("--figure_of_merit", type=str, help="defaults to accuracy for classification or mean_squared_error for regression; should be a value returned from --HuggingFace_metric", default=None)

    args = parser.parse_args()
    wrapped_args = { "wrapper" : args }

    print('calling gft_predict with args: ' + str(args), file=sys.stderr)

    model_provider,model_key = parse_model_specification(wrapped_args)
    task_provider,task_key = parse_task_specification(wrapped_args)

    if task_provider == 'PaddleHub' or model_provider == 'PaddleHub':
        from gft.gft_internals import gft_predict_pd
        return gft_predict_pd.gft_predict_pd(wrapped_args)
    else:
        from gft.gft_internals import gft_predict_hf
        return gft_predict_hf.gft_predict_hf(wrapped_args)

if __name__ == "__main__":
    main()
