#!/bin/sh

echo hostname = `hostname`

gft_fit --model H:bert-base-cased \
    --data H:snli \
    --output_dir $1 \
    --eqn 'classify: label ~ premise + hypothesis' \
    --per_device_train_batch_size=8 \
    --num_train_epochs 20

