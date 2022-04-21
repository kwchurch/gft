#!/bin/sh

echo hostname = `hostname`

gft_predict --model H:bert-base-cased \
    --data H:conll2003 \
    --output_dir $1 \
    --eqn 'classify_tokens: pos_tags ~ tokens' \
    --figure_of_merit overall_accuracy \
    --is_split_into_words \
    --num_train_epochs 20

