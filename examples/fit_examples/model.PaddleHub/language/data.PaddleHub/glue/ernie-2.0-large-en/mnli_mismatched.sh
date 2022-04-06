#!/bin/sh

echo hostname = `hostname`

gft_fit --model P:ernie-2.0-large-en \
    --data P:glue,mnli \
    --metric H:glue,mnli \
    --splits train,dev_mismatched,test_mismatched \
    --output_dir $1 \
    --eqn 'classify: labels ~ sentence1 + sentence2' \
    --per_device_train_batch_size 512 \
    --learning_rate "3e-5" \
    --num_train_epochs 3

