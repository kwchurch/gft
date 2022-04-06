#!/bin/sh
echo hostname = `hostname`

for model in AdapterHub/bert-base-uncased-pf-sick AdapterHub/roberta-base-pf-sick
	     do
data=sick
gft_eval --data H:sick --model H:AdapterHub/bert-base-uncased-pf-sick --split test  --eqn 'classify: label ~ sentence_A + sentence_B'
done
