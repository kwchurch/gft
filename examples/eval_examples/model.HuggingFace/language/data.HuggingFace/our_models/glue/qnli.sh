#!/bin/sh

echo hostname = `hostname`
m=model.HuggingFace
d=data.HuggingFace

gft_eval --model C:$gft_checkpoints/fit_examples/$m/language/$d/glue/qnli/ckpt/best/ \
	 --base_model H:bert-base-cased \
	 --data H:glue,qnli \
	 --metric H:glue,qnli \
	 --eqn 'classify: label ~ question + sentence' \
	 --split val

