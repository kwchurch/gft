#!/bin/sh

echo hostname = `hostname`

subtask=qnli

find /mnt/storage/svail-0/yuchenbian/torch_KD/torch_distill_pudb/output/GLUE_grid/$subtask -name 'pytorch_model.bin' | 
    while read b
    do
	model=`dirname $b`
gft_eval --model C:$model \
	 --base_model H:roberta-large \
	 --data H:glue,$subtask \
	 --metric H:glue,$subtask \
	 --eqn 'classify: label ~ question + sentence' \
	 --split val
done
