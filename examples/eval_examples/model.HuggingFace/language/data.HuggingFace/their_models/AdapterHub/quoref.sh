#!/bin/sh
echo hostname = `hostname`

# multiple choice

for model in AdapterHub/bert-base-uncased-pf-quoref AdapterHub/roberta-base-pf-quoref 
	     do
data=quoref
gft_eval --data H:quoref --model H:AdapterHub/bert-base-uncased-pf-quoref --split test  --eqn 'classify_spans: answers ~ question + context'
done 
