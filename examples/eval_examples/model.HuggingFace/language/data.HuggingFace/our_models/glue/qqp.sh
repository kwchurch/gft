#!/bin/sh

echo hostname = `hostname`
m=model.HuggingFace
d=data.HuggingFace

subtask=qqp

gft_eval --model C:$gft_checkpoints/fit_examples/$m/language/$d/glue/$subtask/ckpt/best/ \
	 --base_model H:bert-base-cased \
	 --data H:glue,$subtask \
	 --metric H:glue,$subtask \
	 --eqn 'classify: label ~ question1 + question2' \
	 --split val

