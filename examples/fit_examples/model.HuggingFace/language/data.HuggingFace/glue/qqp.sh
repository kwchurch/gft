#!/bin/sh

echo hostname = `hostname`

gft_fit --model H:bert-base-cased \
    --data H:glue,qqp \
    --metric H:glue,qqp \
    --output_dir $1 \
    --eqn 'classify: label ~ question1 + question2' \
    --num_train_epochs 3


