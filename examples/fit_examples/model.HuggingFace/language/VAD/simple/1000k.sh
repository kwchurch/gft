#!/bin/sh

echo hostname = `hostname`

gft_fit --model H:bert-base-cased \
    --data C:$gft/datasets/VAD/simple/VAD.simple.1000k \
    --output_dir $1 \
    --eqn 'regress: gold ~ word1 + word2' \
    --num_train_epochs 10
