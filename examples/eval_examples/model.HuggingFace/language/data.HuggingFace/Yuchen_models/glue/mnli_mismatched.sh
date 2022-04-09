#!/bin/sh

echo hostname = `hostname`

subtask=mnli

find /mnt/storage/svail-0/yuchenbian/torch_KD/torch_distill_pudb/output/GLUE_grid/$subtask -name 'pytorch_model.bin' | 
    while read b
    do
	model=`dirname $b`

gft_eval --model C:$model \
	 --base_model H:roberta-large \
	 --data H:glue,mnli \
	 --metric H:glue,mnli \
	 --eqn 'classify: label ~ premise + hypothesis' \
	 --split validation_mismatched
    done
