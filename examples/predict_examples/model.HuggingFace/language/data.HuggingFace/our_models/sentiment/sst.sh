#!/bin/sh

task=sst

model=$gft_checkpoints/fit_examples/model.HuggingFace/language/data.HuggingFace/sentiment/$task/ckpt/best
gft_predict --model H:$model --data H:$task --eqn 'regress: label ~ sentence' --split test # > $model/predict.out 2>$model/predict.err


