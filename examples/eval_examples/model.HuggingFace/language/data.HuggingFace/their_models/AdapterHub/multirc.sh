#!/bin/sh
echo hostname = `hostname`

model=AdapterHub/bert-base-uncased-pf-multirc
data=multirc
gft_eval --data H:multirc --model H:AdapterHub/bert-base-uncased-pf-multirc --split test 
