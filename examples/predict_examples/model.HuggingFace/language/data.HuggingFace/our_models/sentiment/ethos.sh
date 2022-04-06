#!/bin/sh

task=ethos

model=$gft_checkpoints/fit_examples/model.HuggingFace/language/data.HuggingFace/sentiment/$task/ckpt/best
gft_predict --model $model --data H:$task,binary --eqn 'classify: label ~ text' --split train > $model/predict.out 2>$model/predict.err

# model=$gft_checkpoints/model.HuggingFace/language/data.HuggingFace/sentiment/$task/ckpt/best
# gft_dataset --data H:$task,binary --eqn 'classify: label ~ text' --split train |
# gft_predict --model $model

