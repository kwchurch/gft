#!/bin/sh

echo hostname = `hostname`

base_model=ernie-1.0
model=$gft_checkpoints/fit_examples/model.PaddleHub/language/data.PaddleHub/sentiment/chnsenticorp/$base_model/ckpt/best

gft_eval --model P:$model \
    --data P:chnsenticorp \
    --split test \
    --eqn 'classify: label ~ text'



