#!/bin/sh
echo hostname = `hostname`

model=AdapterHub/bert-base-uncased-pf-anli_r3
data=anli_r3
gft_eval --data H:anli --model H:AdapterHub/bert-base-uncased-pf-anli_r3 --split test_r3  --eqn 'classify: label ~ hypothesis + reason'
