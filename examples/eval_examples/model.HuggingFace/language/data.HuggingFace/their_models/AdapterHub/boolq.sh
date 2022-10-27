#!/bin/sh
echo hostname = `hostname`

for model in AdapterHub/bert-base-uncased-pf-boolq AdapterHub/roberta-base-pf-boolq # @ukp/bert-base-uncased-boolq_pfeiffer @ukp/distilbert-base-uncased_qa_boolq_pfeiffer @ukp/roberta-base-boolq_pfeiffer 
	     do
data=boolq
gft_eval --data H:boolq --model H:$model --split val --eqn 'classify: answer ~ question + passage' --debug
done

