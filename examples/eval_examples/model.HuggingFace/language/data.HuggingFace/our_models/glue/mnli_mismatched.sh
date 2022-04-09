#!/bin/sh

echo hostname = `hostname`
m=model.HuggingFace
d=data.HuggingFace

gft_eval --model C:$gft_checkpoints/fit_examples/$m/language/$d/glue/mnli_mismatched/ckpt/best/ \
	 --base_model H:bert-base-cased \
	 --data H:glue,mnli \
	 --metric H:glue,mnli \
	 --eqn 'classify: label ~ premise + hypothesis' \
	 --split validation_mismatched




