#!/bin/sh
echo hostname = `hostname`

model=AdapterHub/bert-base-uncased-pf-mrpc
data=mrpc

# gft_eval --data H:mrpc --model H:AdapterHub/bert-base-uncased-pf-mrpc --split test 

gft_eval --data H:glue,mrpc --eqn 'classify: label ~ sentence1 + sentence2' --split test  --model $model --metric H:glue,mrpc
