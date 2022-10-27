#!/bin/sh
echo hostname = `hostname`

model=AdapterHub/bert-base-uncased-pf-wic
data=wic
gft_eval --data H:wic --model H:AdapterHub/bert-base-uncased-pf-wic --split test 
