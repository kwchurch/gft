#!/bin/sh
echo hostname = `hostname`

model=AdapterHub/bert-base-uncased-pf-conll2003_pos
data=conll2003
gft_eval --data H:conll2003 \
	 --model H:AdapterHub/bert-base-uncased-pf-conll2003_pos \
	 --split test \
	 --eqn 'classify_tokens: pos_tags ~ tokens' \
	 --is_split_into_words

