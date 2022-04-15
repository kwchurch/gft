#!/bin/sh

echo hostname = `hostname`

c=stance_hillary

gft_eval --model C:$gft_checkpoints/fit_examples/model.HuggingFace/language/data.HuggingFace/text_classification/tweet_eval/tweet_eval_$c/ckpt/best \
	 --base_model H:bert-base-cased \
	 --data H:tweet_eval,$c \
	 --eqn 'classify: label ~ text' \
	 --split test

