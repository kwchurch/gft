#!/bin/sh

echo hostname = `hostname`

gft_fit --model P:ernie-2.0-large-en \
    --data P:glue,qqp \
    --metric H:glue,qqp \
    --output_dir $1 \
    --eqn 'classify: labels ~ sentence1 + sentence2' \
    --per_device_train_batch_size 256 \
    --learning_rate "5e-5" \
    --num_train_epochs 3


