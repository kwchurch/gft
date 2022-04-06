#!/bin/sh
echo hostname = `hostname`

for model in AdapterHub/bert-base-uncased-pf-drop AdapterHub/roberta-base-pf-drop
	     do
data=drop
gft_eval --data H:drop --model H:AdapterHub/bert-base-uncased-pf-drop --split test --eqn 'classify_spans: answer_spans ~ question + passage'
done
