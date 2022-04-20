#!/bin/sh

task=poem_sentiment

model=$gft_checkpoints/fit_examples/model.HuggingFace/language/data.HuggingFace/sentiment/$task/ckpt/best
gft_predict --model H:$model --data H:$task --eqn 'classify: label ~ verse_text' --split test # > $model/predict.out 2>$model/predict.err

