#!/bin/sh

echo hostname = `hostname`
. $params

gft_fit --model P:$model \
    --data P:glue,cola \
    --metric H:glue,cola \
    --figure_of_merit matthews_correlation \
    --output_dir $1 \
    --eqn 'classify: labels ~ sentence' \
    --learning_rate $lr \
--per_device_train_batch_size $bs \
--num_train_epochs $epochs

