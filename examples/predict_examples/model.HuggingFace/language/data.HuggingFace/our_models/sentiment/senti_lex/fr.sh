#!/bin/sh

task=senti_lex
subtask=fr

model=$gft_checkpoints/fit_examples/model.HuggingFace/language/data.HuggingFace/sentiment/$task/$subtask/ckpt/best
gft_predict --model $model --data H:$task,$subtask --eqn 'classify: sentiment ~ word' --split train --data_dir "$HOME/.manual_dirs/senti_lex" >$model/predict.out 2>$model/predict.err

