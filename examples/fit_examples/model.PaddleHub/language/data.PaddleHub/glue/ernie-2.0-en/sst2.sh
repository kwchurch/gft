#!/bin/sh

echo hostname = `hostname`

gft_fit --model P:ernie-2.0-en \
    --data P:glue,sst-2 \
    --metric H:glue,sst2 \
    --output_dir $1 \
    --eqn 'classify: labels ~ sentence' \
    --per_device_train_batch_size 256 \
    --learning_rate "2e-5" \
    --num_train_epochs 3

