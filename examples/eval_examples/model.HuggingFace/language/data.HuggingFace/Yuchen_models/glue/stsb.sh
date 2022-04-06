#!/bin/sh

echo hostname = `hostname`

subtask=stsb

for model in /mnt/storage/svail-0/yuchenbian/torch_KD/torch_distill_pudb/output/GLUE_grid/$subtask/roberta-large-hf_baseline/"$subtask"_*
	     do
gft_eval --model C:$model \
	 --data H:glue,$subtask \
	 --metric H:glue,$subtask \
	 --eqn 'regress: label ~ sentence1 + sentence2' \
	 --figure_of_merit pearson \
	 --split val
done
