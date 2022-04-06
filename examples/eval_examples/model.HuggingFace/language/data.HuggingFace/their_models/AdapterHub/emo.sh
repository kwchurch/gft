#!/bin/sh
echo hostname = `hostname`

model=AdapterHub/bert-base-uncased-pf-emo
data=emo
gft_eval --data H:emo --model H:AdapterHub/bert-base-uncased-pf-emo --split test 
