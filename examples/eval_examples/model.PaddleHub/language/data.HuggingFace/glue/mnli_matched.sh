#!/bin/sh

echo hostname = `hostname`
m=model.PaddleHub
d=data.HuggingFace

gft_eval --model P:$gft_checkpoints/fit_examples/$m/language/$d/glue/mnli_matched/ckpt/best/ \
	 --base_model H:bert-base-cased \
	 --data H:glue,mnli \
	 --metric H:glue,mnli \
	 --eqn 'classify: label ~ premise + hypothesis' \
	 --split validation_matched \
	 --do_not_catch_errors


