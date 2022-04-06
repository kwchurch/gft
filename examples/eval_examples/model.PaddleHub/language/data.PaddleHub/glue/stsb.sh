#!/bin/sh

echo hostname = `hostname`

subtask=stsb
m=model.PaddleHub
d=data.PaddleHub

gft_eval --model P:$gft_checkpoints/fit_examples/$m/language/$d/glue/$subtask/ckpt/best/ \
	 --data P:glue,sts-b \
	 --metric H:glue,$subtask \
	 --eqn 'classify: labels ~ sentence1 + sentence2' \
	 --split val 

