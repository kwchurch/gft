#!/bin/sh

echo hostname = `hostname`

gft_predict --model P:bert-base-cased \
    --data H:emotion \
    --output_dir $1 \
    --eqn 'classify: label ~ text' \
    --num_train_epochs 3




