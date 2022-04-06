#!/bin/sh

echo hostname = `hostname`

gft_fit --model H:bert-base-cased \
    --data H:imdb \
    --splits train,__select_from_train__,test \
    --output_dir $1 \
    --eqn 'classify: label ~ text' \
    --num_train_epochs 3


