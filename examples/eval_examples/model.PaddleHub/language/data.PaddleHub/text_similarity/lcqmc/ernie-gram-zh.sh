#!/bin/sh

echo hostname = `hostname`

base_model=ernie-gram-zh
data=lcqmc
model=$gft_checkpoints/fit_examples/model.PaddleHub/language/data.PaddleHub/text_similarity/$data/$base_model/ckpt/best

gft_eval --model P:$base_model \
    --data P:$data \
    --split test \
    --eqn 'classify: label ~ query+title' \
    --do_not_catch_errors

