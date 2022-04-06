#!/bin/sh

echo hostname = `hostname`

pos=fallows
b=16
m=model.HuggingFace

gft_eval --model $gft_checkpoints/fit_examples/$m/language/syn_ant/$pos.$b/ckpt/$b/best \
	 --base_model H:bert-base-cased \
	 --data C:$gft/datasets/syn_ant/tag-$pos-pairs \
	 --eqn 'classify: gold ~ word1 + word2' \
	 --split test \
	 --do_not_catch_errors






