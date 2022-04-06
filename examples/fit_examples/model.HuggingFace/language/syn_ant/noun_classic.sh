#!/bin/sh

echo hostname = `hostname`

gft_fit --model H:bert-base-cased \
    --data C:$gft/datasets/syn_ant/tag-noun-pairs \
    --output_dir $1 \
    --eqn 'classify_classic: gold ~ word1 + word2' \
    --num_train_epochs 10

