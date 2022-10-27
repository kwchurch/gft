#!/bin/sh
echo hostname = `hostname`

model=AdapterHub/bert-base-uncased-pf-scicite
data=scicite
gft_eval --data H:scicite --model H:AdapterHub/bert-base-uncased-pf-scicite --split test 
