#!/bin/sh

echo hostname = `hostname`

subtask=qnli

for model in /mnt/storage/svail-0/yuchenbian/torch_KD/torch_distill_pudb/output/GLUE_grid/$subtask/roberta-large-hf_baseline/"$subtask"_*
	     do
gft_eval --model C:$model \
	 --data H:glue,$subtask \
	 --metric H:glue,$subtask \
	 --eqn 'classify: label ~ question + sentence' \
	 --split val
done
