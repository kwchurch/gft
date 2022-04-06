#!/bin/sh

echo hostname = `hostname`

pos=adj
b=32
m=model.HuggingFace

gft_eval --model $gft_checkpoints/fit_examples/$m/language/syn_ant/$pos.$b/ckpt/$b/best \
	 --base_model H:bert-base-cased \
	 --data C:$gft/datasets/syn_ant/tag-$pos-pairs \
	 --eqn 'classify: gold ~ word1 + word2' \
	 --split test





