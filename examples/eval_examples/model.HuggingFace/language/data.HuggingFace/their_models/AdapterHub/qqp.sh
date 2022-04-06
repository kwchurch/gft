#!/bin/sh
echo hostname = `hostname`

model=AdapterHub/bert-base-uncased-pf-qqp
task=qqp
# gft_eval --data H:qqp --model H:AdapterHub/bert-base-uncased-pf-qqp --split test 
gft_eval --data H:glue,$task --eqn 'classify: label ~ question1 + question2' --split val --model $model --metric H:glue,qqp
