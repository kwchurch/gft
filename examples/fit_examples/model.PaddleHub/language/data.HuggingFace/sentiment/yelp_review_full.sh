#!/bin/sh

echo hostname = `hostname`

# batch size of 64 ran out of memory

# H:yelp_review_full gets a checksum error at HuggingFace

gft_fit --model P:bert-base-cased \
    --data H:yelp_review_full \
    --splits train,__select_from_train__,test \
    --output_dir $1 \
    --eqn 'classify: label ~ text' \
    --per_device_eval_batch_size 32 \
    --per_device_train_batch_size 32 \
    --num_train_epochs 10

