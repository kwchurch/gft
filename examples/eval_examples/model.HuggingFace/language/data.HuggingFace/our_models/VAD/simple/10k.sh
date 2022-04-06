#!/bin/sh

echo hostname = `hostname`
m=model.HuggingFace

gft_eval --model H:$gft_checkpoints/fit_examples/$m/language/VAD/simple/10k/ckpt/best \
    --base_model H:bert-base-cased \
    --data C:$gft/datasets/VAD/simple/VAD.simple.10k \
    --eqn 'regress: gold ~ word1 + word2' \
    --split test


