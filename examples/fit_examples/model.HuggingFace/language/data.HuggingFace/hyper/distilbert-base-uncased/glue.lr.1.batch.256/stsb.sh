#!/bin/sh

echo hostname = `hostname`
. $params

gft_fit --model H:$model \
    --data H:glue,stsb \
    --metric H:glue,stsb \
    --output_dir $1 \
    --eqn 'regress: label ~ sentence1 + sentence2' \
    --figure_of_merit pearson \
    --learning_rate $lr \
    --per_device_eval_batch_size $bs \
    --num_train_epochs 10



