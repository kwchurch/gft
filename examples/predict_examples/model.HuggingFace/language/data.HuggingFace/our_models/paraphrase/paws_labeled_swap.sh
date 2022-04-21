#!/bin/sh

echo hostname = `hostname`

b=4

gft_predict --model H:eugenesiow/bart-paraphrase \
    --data H:paws,labeled_swap \
    --output_dir $1 \
    --eqn 'classify: label ~ sentence1 + sentence2' \
    --splits 'train,__select_from_train__,train' \
    --per_device_train_batch_size $b \
    --per_device_eval_batch_size $b \
    --num_train_epochs 3

