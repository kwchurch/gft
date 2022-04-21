#!/bin/sh

echo hostname = `hostname`

gft_predict --model P:bert-base-cased \
    --data H:banking77 \
    --splits train,__select_from_train__,test \
    --output_dir $1 \
    --eqn 'classify: label ~ text' \
    --num_train_epochs 10

