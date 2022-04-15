#!/bin/sh

echo hostname = `hostname`

subtask=cola
m=model.PaddleHub
d=data.PaddleHub

gft_eval --model P:$gft_checkpoints/fit_examples/$m/language/$d/glue/$subtask/ckpt/best/ \
	 --data P:glue,$subtask \
	 --metric H:glue,$subtask \
	 --figure_of_merit matthews_correlation \
	 --eqn 'classify: labels ~ sentence' \
	 --split val 



