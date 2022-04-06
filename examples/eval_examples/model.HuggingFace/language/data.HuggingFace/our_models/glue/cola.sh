#!/bin/sh

echo hostname = `hostname`

subtask=cola
m=model.HuggingFace
d=data.HuggingFace

gft_eval --model C:$gft_checkpoints/fit_examples/$m/language/$d/glue/$subtask/ckpt/best/ \
	 --base_model H:bert-base-cased \
	 --data H:glue,$subtask \
	 --metric H:glue,$subtask \
	 --figure_of_merit matthews_correlation \
	 --eqn 'classify: label ~ sentence' \
	 --split val







