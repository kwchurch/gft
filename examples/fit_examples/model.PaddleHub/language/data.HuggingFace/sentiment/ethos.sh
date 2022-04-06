#!/bin/sh

echo hostname = `hostname`

gft_fit --model P:bert-base-uncased \
    --data H:ethos,binary \
    --splits train,__select_from_train__,train \
    --output_dir $1 \
    --eqn 'classify: label ~ text' \
    --num_train_epochs 40

