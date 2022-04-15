#!/bin/sh

echo hostname = `hostname`
. $params

gft_fit --model P:$model \
    --data P:glue,qqp \
    --metric H:glue,qqp \
    --output_dir $1 \
    --eqn 'classify: labels ~ sentence1 + sentence2' \
    --learning_rate $lr \
    --per_device_eval_batch_size $bs \
    --num_train_epochs 10

