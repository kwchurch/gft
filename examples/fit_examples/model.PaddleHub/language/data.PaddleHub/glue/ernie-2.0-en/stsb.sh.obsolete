#!/bin/sh

echo hostname = `hostname`

gft_fit --model P:ernie-2.0-en \
    --data P:glue,sts-b \
    --metric H:glue,stsb \
    --output_dir $1 \
    --eqn 'regress: labels ~ sentence1 + sentence2' \
    --figure_of_merit pearson \
    --per_device_train_batch_size 128 \
    --learning_rate "5e-5" \
    --num_train_epochs 3


