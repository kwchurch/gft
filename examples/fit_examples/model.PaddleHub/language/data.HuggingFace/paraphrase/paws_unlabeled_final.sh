#!/bin/sh

echo hostname = `hostname`

b=2

gft_fit --model P:bart-base \
    --data H:paws,unlabeled_final \
    --output_dir $1 \
    --eqn 'classify: label ~ sentence1 + sentence2' \
    --splits 'train,validation,validation' \
    --per_device_train_batch_size $b \
    --num_train_epochs 3

