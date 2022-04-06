#!/bin/sh

echo hostname = `hostname`

gft_eval --model P:bert-base-cased \
    --data H:conll2003 \
    --eqn 'classify_tokens: chunk_tags ~ tokens' \
    --figure_of_merit overall_accuracy \
    --is_split_into_words


