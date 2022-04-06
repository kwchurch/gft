#!/bin/sh

task=imdb

model=$gft_checkpoints/model.PaddleHub/language/data.HuggingFace/sentiment/$task/ckpt/best

gft_predict --data H:$task --eqn 'classify: label ~ text' --split test --model $model

