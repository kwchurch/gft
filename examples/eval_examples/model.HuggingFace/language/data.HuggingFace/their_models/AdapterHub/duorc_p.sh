#!/bin/sh
echo hostname = `hostname`

model=AdapterHub/bert-base-uncased-pf-duorc_p
data=duorc_p
gft_eval --data H:duorc_p --model H:AdapterHub/bert-base-uncased-pf-duorc_p --split test 
