#!/bin/sh

echo hostname = `hostname`
m=model.HuggingFace
d=data.HuggingFace

gft_eval --model C:$gft_checkpoints/fit_examples/$m/language/$d/glue/mrpc/ckpt/best/ \
	 --base_model H:bert-base-cased \
	 --data H:glue,mrpc \
	 --metric H:glue,mrpc \
	 --eqn 'classify: label ~ sentence1 + sentence2' \
	 --split val


