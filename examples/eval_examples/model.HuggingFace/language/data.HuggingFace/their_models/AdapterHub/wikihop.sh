#!/bin/sh
echo hostname = `hostname`

model=AdapterHub/bert-base-uncased-pf-wikihop
data=wikihop
gft_eval --data H:wikihop --model H:AdapterHub/bert-base-uncased-pf-wikihop --split test 
