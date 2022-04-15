#!/bin/sh

echo hostname = `hostname`
. $params

gft_fit --model P:$model \
    --data P:glue,sst-2 \
    --metric H:glue,sst2 \
    --output_dir $1 \
    --eqn 'classify: labels ~ sentence' \
    --learning_rate $lr \
    --per_device_eval_batch_size $bs \
    --num_train_epochs 10

