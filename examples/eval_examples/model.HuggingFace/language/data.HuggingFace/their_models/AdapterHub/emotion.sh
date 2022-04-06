#!/bin/sh
echo hostname = `hostname`

for model in AdapterHub/bert-base-uncased-pf-emotion AdapterHub/roberta-base-pf-emotion
	     do
data=emotion
# gft_eval --data H:emotion --model H:AdapterHub/bert-base-uncased-pf-emotion --split test 

gft_eval --model H:$model \
    --data H:emotion \
    --eqn 'classify: label ~ text' \
    --split test
done
