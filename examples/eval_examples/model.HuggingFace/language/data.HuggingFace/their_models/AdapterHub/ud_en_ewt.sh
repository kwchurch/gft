#!/bin/sh
echo hostname = `hostname`

model=AdapterHub/bert-base-uncased-pf-ud_en_ewt
data=ud_en_ewt
gft_eval --data H:ud_en_ewt --model H:AdapterHub/bert-base-uncased-pf-ud_en_ewt --split test 
