#!/bin/sh
echo hostname = `hostname`

model=AdapterHub/bert-base-uncased-pf-ud_deprel
data=ud_deprel
gft_eval --data H:ud_deprel --model H:AdapterHub/bert-base-uncased-pf-ud_deprel --split test 
