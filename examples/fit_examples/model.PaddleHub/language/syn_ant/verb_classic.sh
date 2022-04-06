#!/bin/sh

echo hostname = `hostname`

gft_fit --model P:bert-base-cased \
    --data C:$gft/datasets/syn_ant/tag-verb-pairs \
    --output_dir $1 \
    --eqn 'classify_classic: gold ~ word1 + word2' \
    --num_train_epochs 10

