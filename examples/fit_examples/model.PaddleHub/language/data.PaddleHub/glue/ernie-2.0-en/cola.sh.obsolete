#!/bin/sh

echo hostname = `hostname`

gft_fit --model P:ernie-2.0-en \
    --data P:glue,cola \
    --metric H:glue,cola \
    --figure_of_merit matthews_correlation \
    --output_dir $1 \
    --eqn 'classify: labels ~ sentence' \
    --per_device_train_batch_size 64 \
    --learning_rate "3e-5" \
    --num_train_epochs 3
