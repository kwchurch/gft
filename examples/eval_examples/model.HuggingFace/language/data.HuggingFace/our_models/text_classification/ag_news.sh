#!/bin/sh

echo hostname = `hostname`

task=ag_news

gft_eval --model C:$gft_checkpoints/fit_examples/model.HuggingFace/language/data.HuggingFace/text_classification/$task/ckpt/best \
	 --base_model H:bert-base-cased \
	 --data H:$task \
	 --eqn 'classify: label ~ text' \
	 --split test


