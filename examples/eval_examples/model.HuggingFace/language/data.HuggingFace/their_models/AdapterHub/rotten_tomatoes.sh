#!/bin/sh
echo hostname = `hostname`

for model in AdapterHub/bert-base-uncased-pf-rotten_tomatoes AdapterHub/roberta-base-pf-rotten_tomatoes
	     do
data=rotten_tomatoes
gft_eval --data H:rotten_tomatoes --model H:AdapterHub/bert-base-uncased-pf-rotten_tomatoes --split test --eqn 'classify: label ~ text'

done
