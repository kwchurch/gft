#!/bin/sh

echo hostname = `hostname`

gft_predict --model P:bert-base-multilingual-cased \
    --data H:amazon_reviews_multi \
    --output_dir $1 \
    --eqn 'classify: stars ~ review_title + review_body' \
    --num_train_epochs 20

