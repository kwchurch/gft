#!/bin/sh

echo hostname = `hostname`

gft_fit --model P:bert-base-cased \
    --data H:tweets_hate_speech_detection \
    --splits 'train,__select_from_train__,train' \
    --output_dir $1 \
    --eqn 'classify: label ~ tweet' \
    --num_train_epochs 10



