#!/bin/sh

echo hostname = `hostname`

gft_fit --model P:ernie-2.0-large-en \
    --data P:chnsenticorp \
    --output_dir $1 \
    --eqn 'classify: label ~ text' \
    --per_device_train_batch_size 24 \
    --learning_rate "1e-5" \
    --num_train_epochs 3


