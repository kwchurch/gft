#!/bin/sh
echo hostname = `hostname`

for model in AdapterHub/bert-base-uncased-pf-hotpotqa AdapterHub/roberta-base-pf-hotpotqa
	     do
gft_eval --data H:hotpotqa --model H:$model --split test --eqn 'classify_spans: answer ~ question + context'
done
