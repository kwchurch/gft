#!/bin/sh

echo hostname = `hostname`

b=64

gft_fit --model P:bert-base-cased \
    --data C:$gft/datasets/syn_ant/tag-fallows-pairs \
    --output_dir $1/$b \
    --eqn 'classify: gold ~ word1 + word2' \
    --per_device_train_batch_size $b \
    --per_device_eval_batch_size $b \
    --num_train_epochs 10



