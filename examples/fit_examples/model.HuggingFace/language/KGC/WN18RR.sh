#!/bin/sh

echo hostname = `hostname`

gft_fit --model H:bert-base-cased \
    --data C:$gft/datasets/KGC/WN18RR \
    --output_dir $1 \
    --eqn 'classify: gold ~ Word1 + Word2' \
    --num_train_epochs 10

