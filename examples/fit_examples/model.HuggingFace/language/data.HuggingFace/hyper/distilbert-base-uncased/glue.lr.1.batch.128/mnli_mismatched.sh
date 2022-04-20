#!/bin/sh

echo hostname = `hostname`
. $params

gft_fit --model H:$model \
    --data H:glue,mnli \
    --metric H:glue,mnli \
    --splits train,validation_mismatched,test_mismatched \
    --output_dir $1 \
    --eqn 'classify: label ~ premise + hypothesis' \
    --learning_rate $lr \
--per_device_train_batch_size $bs \
    --num_train_epochs 10

