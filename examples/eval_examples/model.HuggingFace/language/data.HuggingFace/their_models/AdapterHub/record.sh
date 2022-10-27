#!/bin/sh
echo hostname = `hostname`

model=AdapterHub/bert-base-uncased-pf-record
data=record
gft_eval --data H:record --model H:AdapterHub/bert-base-uncased-pf-record --split test 
