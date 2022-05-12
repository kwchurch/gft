#!/bin/sh

model=$gft_checkpoints/fit_examples/model.PaddleHub/language/KGC/WN18RR/ckpt/best

echo hostname = `hostname`

gft_predict --model P:$model \
    --data C:$gft/datasets/KGC/WN18RR \
    --eqn 'classify: gold ~ Word1 + Word2' \
    --split test


