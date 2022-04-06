#!/bin/sh

echo hostname = `hostname`

gft_fit --model H:bert-base-cased \
    --data H:emotion \
    --output_dir $1 \
    --eqn 'classify: label ~ text' \
    --num_train_epochs 20 \
    --num_train_epochs 10



