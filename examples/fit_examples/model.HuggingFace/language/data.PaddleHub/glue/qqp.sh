#!/bin/sh

echo hostname = `hostname`

gft_fit --model H:bert-base-cased \
    --data P:glue,qqp \
    --metric H:glue,qqp \
    --output_dir $1 \
    --eqn 'classify: labels ~ sentence1 + sentence2' \
    --num_train_epochs 3


