#!/bin/sh

echo hostname = `hostname`

gft_fit --model P:bert-base-cased \
    --data H:glue,mnli \
    --metric H:glue,mnli \
    --splits train,validation_matched,test_matched \
    --output_dir $1 \
    --eqn 'classify: label ~ premise + hypothesis' \
    --num_train_epochs 3

