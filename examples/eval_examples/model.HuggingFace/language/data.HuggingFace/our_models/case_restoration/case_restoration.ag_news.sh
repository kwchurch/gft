#!/bin/sh

ds=ag_news

dir=$gft_checkpoints/fit_examples/model.HuggingFace/language/data.HuggingFace/case_restoration/case_restoration.$ds/ckpt
gft_eval --model H:$dir/best \
	 --data C:$dir/$ds \
	 --eqn 'classify_tokens: case_labels ~ tokens' \
	 --figure_of_merit overall_accuracy

