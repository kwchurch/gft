#!/bin/sh
echo hostname = `hostname`

for model in AdapterHub/bert-base-uncased-pf-trec AdapterHub/roberta-base-pf-trec
	     do
data=trec
gft_eval --data H:trec --model H:AdapterHub/bert-base-uncased-pf-trec --split test  --eqn 'classify: label-fine ~ text'
done
