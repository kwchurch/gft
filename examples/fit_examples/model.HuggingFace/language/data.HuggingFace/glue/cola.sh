#!/bin/sh

echo hostname = `hostname`

gft_fit --model H:bert-base-cased \
    --data H:glue,cola \
    --metric H:glue,cola \
    --figure_of_merit matthews_correlation \
    --output_dir $1 \
    --eqn 'classify: label ~ sentence' \
    --num_train_epochs 3



