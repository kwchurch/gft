#!/bin/sh

echo hostname = `hostname`

gft_fit --model P:skep_ernie_1.0_large_ch \
    --data P:lcqmc \
    --output_dir $1 \
    --eqn 'classify: label ~ query+title' \
    --num_train_epochs 3


