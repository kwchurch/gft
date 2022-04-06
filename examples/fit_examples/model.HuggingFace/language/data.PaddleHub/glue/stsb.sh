#!/bin/sh

echo hostname = `hostname`

gft_fit --model H:bert-base-cased \
    --data P:glue,sts-b \
    --metric H:glue,stsb \
    --output_dir $1 \
    --eqn 'regress: labels ~ sentence1 + sentence2' \
    --figure_of_merit pearson \
    --num_train_epochs 3


