#!/bin/sh

echo hostname = `hostname`
. $params


gft_fit --model H:$model \
    --data H:glue,sst2 \
    --metric H:glue,sst2 \
    --output_dir $1 \
    --eqn 'classify: label ~ sentence' \
    --learning_rate $lr \
    --per_device_eval_batch_size $bs \
    --num_train_epochs 10

