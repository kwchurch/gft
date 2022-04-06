#!/bin/sh

echo hostname = `hostname`

gft_fit --model H:bert-base-cased \
    --data H:SetFit/amazon_polarity \
    --splits train,__select_from_train__,test \
    --output_dir $1 \
    --eqn 'classify: label_text ~ title + text' \
    --num_train_epochs 3

