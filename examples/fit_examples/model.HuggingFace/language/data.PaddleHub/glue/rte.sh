#!/bin/sh

echo hostname = `hostname`

gft_fit --model H:bert-base-cased \
    --data P:glue,rte \
    --metric H:glue,rte \
    --output_dir $1 \
    --eqn 'classify: labels ~ sentence1 + sentence2' \
    --num_train_epochs 3


