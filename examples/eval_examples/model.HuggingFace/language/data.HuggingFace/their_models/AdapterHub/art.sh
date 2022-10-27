#!/bin/sh
echo hostname = `hostname`

model=AdapterHub/bert-base-uncased-pf-art
data=art
gft_eval --data H:art --model H:AdapterHub/bert-base-uncased-pf-art --split test 
