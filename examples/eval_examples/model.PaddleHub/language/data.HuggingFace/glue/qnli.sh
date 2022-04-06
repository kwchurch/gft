#!/bin/sh

echo hostname = `hostname`
m=model.PaddleHub
d=data.HuggingFace

gft_eval --model P:$gft_checkpoints/fit_examples/$m/language/$d/glue/qnli/ckpt/best/ \
	 --base_model H:bert-base-cased \
	 --data H:glue,qnli \
	 --metric H:glue,qnli \
	 --eqn 'classify: label ~ question + sentence' \
	 --split val \
	 --do_not_catch_errors

