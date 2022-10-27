#!/bin/sh
echo hostname = `hostname`

for model in AdapterHub/bert-base-uncased-pf-squad_v2 AdapterHub/roberta-base-pf-squad_v2
	     do
data=squad_v2
# gft_eval --data H:squad_v2 --model H:AdapterHub/bert-base-uncased-pf-squad_v2 --split test 


b=8

gft_eval --model H:$model \
    --data H:squad_v2 \
    --metric H:squad \
    --splits train,validation,validation \
    --split validation \
    --eqn 'classify_spans: answers ~ question + context'

done
