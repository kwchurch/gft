#!/bin/sh

echo hostname = `hostname`

gft_fit --model P:bert-base-cased \
    --data H:glue,mrpc \
    --metric H:glue,mrpc \
    --output_dir $1 \
    --eqn 'classify: label ~ sentence1 + sentence2' \
    --num_train_epochs 3

