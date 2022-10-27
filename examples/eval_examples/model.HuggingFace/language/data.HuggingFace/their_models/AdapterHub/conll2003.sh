#!/bin/sh
echo hostname = `hostname`

for model in AdapterHub/bert-base-uncased-pf-conll2003 AdapterHub/roberta-base-pf-conll2003
	     do
data=conll2003
gft_eval --data H:conll2003 --model H:$model --split test 
done
