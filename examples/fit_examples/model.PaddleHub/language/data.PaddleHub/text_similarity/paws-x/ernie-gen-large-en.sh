#!/bin/sh

echo hostname = `hostname`

gft_fit --model P:ernie-gen-large-en \
    --data P:paws-x \
    --output_dir $1 \
    --eqn 'classify: label ~ sentence1+sentence2' \
    --num_train_epochs 3


