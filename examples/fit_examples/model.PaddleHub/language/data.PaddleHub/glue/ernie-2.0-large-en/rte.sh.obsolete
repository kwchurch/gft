#!/bin/sh

echo hostname = `hostname`

gft_fit --model P:ernie-2.0-large-en \
    --data P:glue,rte \
    --metric H:glue,rte \
    --output_dir $1 \
    --eqn 'classify: labels ~ sentence1 + sentence2' \
    --per_device_train_batch_size 4 \
    --learning_rate "3e-5" \
    --num_train_epochs 3


