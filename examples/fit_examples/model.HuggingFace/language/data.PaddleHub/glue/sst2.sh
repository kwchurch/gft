#!/bin/sh

echo hostname = `hostname`

gft_fit --model H:bert-base-cased \
    --data P:glue,sst-2 \
    --metric H:glue,sst2 \
    --output_dir $1 \
    --eqn 'classify: labels ~ sentence' \
    --num_train_epochs 3

