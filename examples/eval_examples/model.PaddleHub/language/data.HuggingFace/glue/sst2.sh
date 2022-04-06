#!/bin/sh

echo hostname = `hostname`
m=model.PaddleHub
d=data.HuggingFace

subtask=sst2

gft_eval --model P:$gft_checkpoints/fit_examples/$m/language/$d/glue/$subtask/ckpt/best/ \
	 --base_model H:bert-base-cased \
	 --data H:glue,$subtask \
	 --metric H:glue,$subtask \
	 --eqn 'classify: label ~ sentence' \
	 --split val \
	 --do_not_catch_errors


