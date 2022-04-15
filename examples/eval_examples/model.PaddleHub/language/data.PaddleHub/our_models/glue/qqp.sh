#!/bin/sh

echo hostname = `hostname`
m=model.PaddleHub
d=data.PaddleHub

subtask=qqp

for model in `find $gft_checkpoints/fit_examples/  -name 'best' | egrep 'model.PaddleHub.*glue' | egrep $subtask`
	     do

gft_eval --model P:$model \
	 --data P:glue,$subtask \
	 --metric H:glue,$subtask \
	 --eqn 'classify: labels ~ sentence1 + sentence2' \
	 --split val

done
