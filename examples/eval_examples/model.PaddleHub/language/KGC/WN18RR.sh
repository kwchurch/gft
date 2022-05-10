#!/bin/sh

echo hostname = `hostname`
m=model.PaddleHub

gft_eval --model H:$gft_checkpoints/fit_examples/$m/language/KGC/WN18RR/ckpt/best \
    --base_model H:bert-base-cased \
    --data C:$gft/datasets/KGC/WN18RR \
    --eqn 'classify: gold ~ Word1 + Word2' \
    --split test

