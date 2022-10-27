#!/bin/sh
echo hostname = `hostname`

model=AdapterHub/bert-base-uncased-pf-rte
task=rte
# gft_eval --data H:rte --model H:AdapterHub/bert-base-uncased-pf-rte --split test 
gft_eval --data H:glue,$task --eqn 'classify: label ~ sentence1 + sentence2' --split val --model $model --metric H:glue,rte
