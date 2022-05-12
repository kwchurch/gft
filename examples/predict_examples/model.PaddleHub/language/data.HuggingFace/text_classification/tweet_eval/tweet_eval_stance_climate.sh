#!/bin/sh

echo hostname = `hostname`

c=stance_climate
model=$gft_checkpoints/fit_examples/model.PaddleHub/language/data.HuggingFace/text_classification/tweet_eval/tweet_eval_$c/ckpt/best

gft_predict --model P:$model \
    --data H:tweet_eval,$c \
    --eqn 'classify: label ~ text' \
    --split test

# gft_fit --model P:bert-base-cased \
#     --data H:tweet_eval,$c \
#     --output_dir $1 \
#     --eqn 'classify: label ~ text' \
#     --num_train_epochs 10


