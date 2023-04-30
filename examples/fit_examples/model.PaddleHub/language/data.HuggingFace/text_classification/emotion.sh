#!/bin/sh

echo hostname = `hostname`

gft_fit --model P:bert-base-cased \
    --data H:dair-ai/emotion \
    --output_dir $1 \
    --eqn 'classify: label ~ text' \
    --num_train_epochs 3




