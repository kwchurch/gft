#!/bin/sh

echo hostname = `hostname`
. $params

gft_fit --model P:$model \
    --data P:glue,sts-b \
    --metric H:glue,stsb \
    --output_dir $1 \
    --eqn 'regress: labels ~ sentence1 + sentence2' \
    --figure_of_merit pearson \
    --learning_rate $lr \
    --per_device_eval_batch_size $bs \
    --num_train_epochs 10


