#!/bin/sh

echo hostname = `hostname`

gft_fit --model P:ernie-gram-zh \
    --data P:lcqmc \
    --output_dir $1 \
    --eqn 'classify: label ~ query+title' \
    --num_train_epochs 3


