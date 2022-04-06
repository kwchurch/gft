#!/bin/sh

echo hostname = `hostname`

b=8

gft_fit --model H:bert-base-cased \
    --data H:squad \
    --metric H:squad \
    --splits train,validation,validation \
    --output_dir $1 \
    --eqn 'classify_spans: answers ~ question + context' \
    --per_device_train_batch_size $b \
    --num_train_epochs 2 \
    --max_seq_length 384 \
    --doc_stride 128 
