#!/bin/sh

echo hostname = `hostname`

c=irony
model=$gft_checkpoints/fit_examples/model.HuggingFace/language/data.HuggingFace/text_classification/tweet_eval/tweet_eval_$c/ckpt/best

gft_predict --model H:$model --base_model H:bert-base-cased \
    --data H:tweet_eval,$c \
    --eqn 'classify: label ~ text' \
    --split test



