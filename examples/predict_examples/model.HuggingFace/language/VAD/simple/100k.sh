#!/bin/sh

echo hostname = `hostname`

gft_predict --model H:bert-base-cased \
    --data C:$gft/datasets/VAD/simple/VAD.simple.100k \
    --output_dir $1 \
    --eqn 'regress: gold ~ word1 + word2' \
    --num_train_epochs 10



