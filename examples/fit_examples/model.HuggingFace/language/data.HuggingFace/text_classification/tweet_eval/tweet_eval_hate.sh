#!/bin/sh

echo hostname = `hostname`

c=hate

gft_fit --model H:bert-base-cased \
    --data H:tweet_eval,$c \
    --output_dir $1 \
    --eqn 'classify: label ~ text' \
    --num_train_epochs 10


