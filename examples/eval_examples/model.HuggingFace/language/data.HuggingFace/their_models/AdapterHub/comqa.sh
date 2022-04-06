#!/bin/sh
echo hostname = `hostname`

model=AdapterHub/bert-base-uncased-pf-comqa
data=comqa
gft_eval --data H:comqa --model H:AdapterHub/bert-base-uncased-pf-comqa --split test 
