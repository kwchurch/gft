#!/bin/sh
echo hostname = `hostname`

model=AdapterHub/bert-base-uncased-pf-pmb_sem_tagging
data=pmb_sem_tagging
gft_eval --data H:pmb_sem_tagging --model H:AdapterHub/bert-base-uncased-pf-pmb_sem_tagging --split test 
