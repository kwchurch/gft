#!/bin/sh

echo hostname = `hostname`

subtask=sst2

for model in /mnt/storage/svail-0/yuchenbian/torch_KD/torch_distill_pudb/output/GLUE_grid/$subtask/roberta-large-hf_baseline/"$subtask"_*
	     do

gft_eval --model C:$model \
	 --data H:glue,$subtask \
	 --metric H:glue,$subtask \
	 --eqn 'classify: label ~ sentence' \
	 --split val
done


