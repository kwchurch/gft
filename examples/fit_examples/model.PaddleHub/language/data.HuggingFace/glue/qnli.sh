#!/bin/sh

echo hostname = `hostname`

gft_fit --model P:bert-base-cased \
    --data H:glue,qnli \
    --metric H:glue,qnli \
    --output_dir $1 \
    --eqn 'classify: label ~ question + sentence' \
    --num_train_epochs 3

