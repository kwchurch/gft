#!/bin/sh

echo hostname = `hostname`

gft_fit --model H:bert-base-cased \
    --data H:glue,sst2 \
    --metric H:glue,sst2 \
    --output_dir $1 \
    --eqn 'classify: label ~ sentence' \
    --num_train_epochs 3

