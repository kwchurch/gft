#!/bin/sh

echo hostname = `hostname`

subtask=mnli
m=model.PaddleHub
d=data.PaddleHub

gft_eval --model P:$gft_checkpoints/fit_examples/$m/language/$d/glue/$subtask/ckpt/best/ \
	 --data P:glue,$subtask \
	 --metric H:glue,$subtask \
	 --splits train,dev_matched,test_matched \
	 --eqn 'classify: labels ~ sentence1 + sentence2' \
	 --split dev_matched


