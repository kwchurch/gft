#!/bin/sh
echo hostname = `hostname`

model=AdapterHub/bert-base-uncased-pf-conll2000
data=conll2000

# echo task = ner

# gft_eval --model H:$model \
#     --data H:$data
#     --eqn 'classify_tokens: ner_tags ~ tokens' \
#     --figure_of_merit overall_accuracy \
#     --is_split_into_words \
#     --split test

echo task = pos

gft_eval --model H:$model \
    --data H:$data
    --eqn 'classify_tokens: pos_tags ~ tokens' \
    --figure_of_merit overall_accuracy \
    --is_split_into_words \
    --split test
