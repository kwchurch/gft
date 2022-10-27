#!/bin/sh
echo hostname = `hostname`

model=AdapterHub/bert-base-uncased-pf-winogrande
data=winogrande
gft_eval --data H:winogrande --model H:AdapterHub/bert-base-uncased-pf-winogrande --split test 
