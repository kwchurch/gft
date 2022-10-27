#!/bin/sh
echo hostname = `hostname`

model=AdapterHub/bert-base-uncased-pf-cq
data=cq
gft_eval --data H:cq --model H:AdapterHub/bert-base-uncased-pf-cq --split test 
