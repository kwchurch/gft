#!/bin/sh

echo hostname = `hostname`
. $params

gft_fit --model H:$model \
    --data H:glue,mnli \
    --metric H:glue,mnli \
    --splits train,validation_matched,test_matched \
    --output_dir $1 \
    --eqn 'classify: label ~ premise + hypothesis' \
    --learning_rate $lr \
    --per_device_eval_batch_size $bs \
    --num_train_epochs 10

