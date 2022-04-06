#!/bin/sh

echo hostname = `hostname`

gft_fit --model H:bert-base-cased \
    --data H:ag_news \
    --splits train,__select_from_train__,test \
    --output_dir $1 \
    --eqn 'classify: label ~ text' \
    --per_device_eval_batch_size 32 \
    --per_device_train_batch_size 32  \
    --num_train_epochs 10

