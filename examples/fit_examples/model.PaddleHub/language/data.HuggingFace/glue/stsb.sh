#!/bin/sh

echo hostname = `hostname`

gft_fit --model P:bert-base-cased \
    --data H:glue,stsb \
    --metric H:glue,stsb \
    --output_dir $1 \
    --eqn 'regress: label ~ sentence1 + sentence2' \
    --figure_of_merit pearson \
    --num_train_epochs 3


