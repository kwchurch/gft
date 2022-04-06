#!/bin/sh

echo hostname = `hostname`

gft_fit --model P:bert-base-cased \
    --data H:snli \
    --output_dir $1 \
    --eqn 'classify: label ~ premise + hypothesis' \
    --num_train_epochs 20

