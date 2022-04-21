#!/bin/sh

echo hostname = `hostname`

b=8

gft_predict --model P:bert-base-cased \
    --data H:squad_v2 \
    --metric H:squad_v2 \
    --splits train,validation,validation \
    --output_dir $1 \
    --eqn 'classify_spans: answers ~ question + context' \
    --per_device_train_batch_size $b \
    --num_train_epochs 2

