#!/bin/sh

echo hostname = `hostname`

# H:amazon_polarity gets a checksum error (needs to be fixed at HuggingFace)

gft_fit --model H:bert-base-cased \
    --data H:amazon_polarity \
    --splits train,__select_from_train__,test \
    --output_dir $1 \
    --eqn 'classify: label ~ title + content' \
    --num_train_epochs 20

