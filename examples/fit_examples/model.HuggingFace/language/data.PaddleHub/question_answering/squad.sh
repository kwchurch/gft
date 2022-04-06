#!/bin/sh

echo hostname = `hostname`

b=8

gft_fit --model H:bert-base-cased \
    --data P:squad \
    --metric H:squad \
    --splits train_v1,dev_v1,dev_v1 \
    --output_dir $1 \
    --eqn 'classify_spans: answers ~ question + context' \
    --per_device_train_batch_size $b \
    --num_train_epochs 2


