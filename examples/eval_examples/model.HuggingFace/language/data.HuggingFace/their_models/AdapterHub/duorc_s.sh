#!/bin/sh
echo hostname = `hostname`

model=AdapterHub/bert-base-uncased-pf-duorc_s
data=duorc_s
gft_eval --data H:duorc_s --model H:AdapterHub/bert-base-uncased-pf-duorc_s --split test 
