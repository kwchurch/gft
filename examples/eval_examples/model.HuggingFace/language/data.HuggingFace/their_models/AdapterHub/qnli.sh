#!/bin/sh
echo hostname = `hostname`

model=AdapterHub/bert-base-uncased-pf-qnli
task=qnli

gft_eval --data H:glue,$task --eqn 'classify: label ~ question + sentence' --split val  --model $model --metric H:glue,$task
