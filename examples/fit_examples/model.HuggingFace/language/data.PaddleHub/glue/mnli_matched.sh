#!/bin/sh

echo hostname = `hostname`

gft_fit --model H:bert-base-cased \
    --data P:glue,mnli \
    --metric H:glue,mnli \
    --splits train,dev_matched,test_matched \
    --output_dir $1 \
    --eqn 'classify: labels ~ sentence1 + sentence2' \
    --num_train_epochs 3


