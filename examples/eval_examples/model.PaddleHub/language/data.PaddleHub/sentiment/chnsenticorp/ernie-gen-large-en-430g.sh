#!/bin/sh

echo hostname = `hostname`

base_model=ernie-gen-large-en-430g
model=$gft_checkpoints/fit_examples/model.PaddleHub/language/data.PaddleHub/sentiment/chnsenticorp/$base_model/ckpt/best

gft_eval --model P:$model \
    --data P:chnsenticorp \
    --eqn 'classify: label ~ text' 


