#!/bin/sh

echo hostname = `hostname`

gft_fit --model P:ernie-tiny \
    --data P:chnsenticorp \
    --output_dir $1 \
    --eqn 'classify: label ~ text' \
    --per_device_train_batch_size 24 \
    --learning_rate "5e-5" \
    --num_train_epochs 3


