#!/bin/sh

task=ethos

model=$gft_checkpoints/fit_examples/model.HuggingFace/language/data.HuggingFace/sentiment/$task/ckpt/best
gft_predict --model H:$model --data H:$task,binary --eqn 'classify: label ~ text' --split train # > $model/predict.out 2>$model/predict.err

