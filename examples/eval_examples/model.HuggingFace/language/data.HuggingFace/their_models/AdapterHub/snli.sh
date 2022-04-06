#!/bin/sh
echo hostname = `hostname`

for model in AdapterHub/bert-base-uncased-pf-snli AdapterHub/roberta-base-pf-snli
	     do

data=snli
# gft_eval --data H:snli --model H:AdapterHub/bert-base-uncased-pf-snli --split test 

gft_eval --model H:$model \
    --data H:snli \
    --eqn 'classify: label ~ premise + hypothesis' \
    --split test
done
