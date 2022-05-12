#!/bin/sh

pos=fallows
b=32
mhub=model.PaddleHub

model=$gft_checkpoints/fit_examples/$mhub/language/syn_ant/$pos.$b/ckpt/$b/best

# model=$gft_checkpoints/fit_examples/model.HuggingFace/language/data.HuggingFace/sentiment/$task/ckpt/best
# gft_predict --model $model --data H:$task,binary --eqn 'classify: label ~ text' --split train > $model/predict.out 2>$model/predict.err


# gft_predict --data C:$gft/datasets/syn_ant/tag-"$pos"-pairs --eqn 'classify_classic: gold ~ word1 + word2' --split test --model P:$model # > $model/predict.out 2>$model/predict.err

gft_predict --model P:$model \
	    --data C:$gft/datasets/syn_ant/tag-"$pos"-pairs \
	    --eqn 'classify_classic: gold ~ word1 + word2' \
	    --split test
