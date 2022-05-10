#!/bin/sh

echo hostname = `hostname`

for model in mrm8488/mobilebert-finetuned-pos # AdapterHub/bert-base-uncased-pf-conll2000
	      do

# gft_eval --model H:$model \
#     --data H:conll2003 \
#     --eqn 'classify_tokens: pos_tags ~ tokens' \
#     --figure_of_merit overall_accuracy \
#     --is_split_into_words \
#     --split test \
#     --do_not_catch_errors

gft_eval --model H:$model \
    --data H:conll2003 \
    --eqn 'classify_tokens: pos_tags ~ tokens' \
    --split test \
    --figure_of_merit overall_accuracy \
    --is_split_into_words \
    --pad_to_max_length

done
