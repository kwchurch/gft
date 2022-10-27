#!/bin/sh
echo hostname = `hostname`

model=AdapterHub/bert-base-uncased-pf-copa
data=copa
gft_eval --data H:copa --model H:AdapterHub/bert-base-uncased-pf-copa --split test 
