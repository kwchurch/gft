#!/bin/sh

echo hostname = `hostname`

subtask=sst2
m=model.PaddleHub
d=data.PaddleHub

gft_eval --model P:$gft_checkpoints/fit_examples/$m/language/$d/glue/$subtask/ckpt/best/ \
	 --data P:glue,sst-2 \
	 --metric H:glue,$subtask \
	 --eqn 'classify: labels ~ sentence' \
	 --split val
