#!/bin/sh

echo hostname = `hostname`

gft_fit --model H:bert-base-cased \
    --data P:glue,cola \
    --metric H:glue,cola \
    --figure_of_merit matthews_correlation \
    --output_dir $1 \
    --eqn 'classify: labels ~ sentence' \
    --num_train_epochs 3



