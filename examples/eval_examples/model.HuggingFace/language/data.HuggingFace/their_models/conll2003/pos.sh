#!/bin/sh

echo hostname = `hostname`

for models in AdapterHub/bert-base-uncased-pf-conll2000
	      do
gft_eval --model H:$models \
    --data H:conll2003 \
    --eqn 'classify_tokens: pos_tags ~ tokens' \
    --figure_of_merit overall_accuracy \
    --is_split_into_words \
    --split test
done



