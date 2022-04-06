#!/bin/sh

task=sst

model=$gft_checkpoints/fit_examples/model.HuggingFace/language/data.HuggingFace/sentiment/$task/ckpt/best
gft_predict --model $model --data H:$task --eqn 'regress: label ~ sentence' --split test > $model/predict.out 2>$model/predict.err

# model=$gft_checkpoints/model.HuggingFace/language/data.HuggingFace/sentiment/$task/ckpt/best

# gft_dataset --data H:$task --eqn 'regress: label ~ sentence' --split test |
# gft_predict --model $model

