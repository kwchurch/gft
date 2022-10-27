#!/bin/sh
echo hostname = `hostname`

model=AdapterHub/bert-base-uncased-pf-stsb
task=stsb
# gft_eval --data H:stsb --model H:AdapterHub/bert-base-uncased-pf-stsb --split test 
gft_eval --data H:glue,$task --eqn 'regress: label ~ sentence1 + sentence2' --split val  --model $model --metric H:glue,stsb --figure_of_merit pearson 
