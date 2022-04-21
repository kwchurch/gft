#!/bin/sh

echo hostname = `hostname`

gft_predict --model P:bert-base-cased \
    --data H:poem_sentiment \
    --output_dir $1 \
    --eqn 'classify: label ~ verse_text' \
    --num_train_epochs 20

