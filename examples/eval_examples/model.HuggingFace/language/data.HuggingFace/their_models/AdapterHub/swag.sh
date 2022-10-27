#!/bin/sh
echo hostname = `hostname`

model=AdapterHub/bert-base-uncased-pf-swag
data=swag
gft_eval --data H:swag --model H:AdapterHub/bert-base-uncased-pf-swag --split test 
