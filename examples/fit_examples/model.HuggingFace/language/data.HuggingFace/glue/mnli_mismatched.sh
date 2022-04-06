#!/bin/sh

echo hostname = `hostname`

gft_fit --model H:bert-base-cased \
    --data H:glue,mnli \
    --metric H:glue,mnli \
    --splits train,validation_mismatched,test_mismatched \
    --output_dir $1 \
    --eqn 'classify: label ~ premise + hypothesis' \
    --num_train_epochs 3

