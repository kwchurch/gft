#!/bin/sh

echo hostname = `hostname`

# batch size of 64 ran out of memory

gft_fit --model H:bert-base-cased \
    --data H:SetFit/yelp_review_full \
    --splits train,__select_from_train__,test \
    --output_dir $1 \
    --eqn 'classify: label_text ~ text' \
    --per_device_eval_batch_size 32 \
    --per_device_train_batch_size 32 \
    --num_train_epochs 3


