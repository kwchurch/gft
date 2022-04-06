#!/bin/sh
echo hostname = `hostname`

model=AdapterHub/bert-base-uncased-pf-wnut_17
data=wnut_17
gft_eval --data H:wnut_17 --model H:AdapterHub/bert-base-uncased-pf-wnut_17 --split test 
