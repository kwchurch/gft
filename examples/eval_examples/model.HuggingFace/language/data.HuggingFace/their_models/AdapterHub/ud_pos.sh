#!/bin/sh
echo hostname = `hostname`

model=AdapterHub/bert-base-uncased-pf-ud_pos
data=ud_pos
gft_eval --data H:ud_pos --model H:AdapterHub/bert-base-uncased-pf-ud_pos --split test 
