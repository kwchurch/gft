#!/bin/sh

echo hostname = `hostname`

gft_fit --model P:bert-base-cased \
    --data H:sst \
    --output_dir $1 \
    --eqn 'regress: label ~ sentence' \
    --num_train_epochs 20

