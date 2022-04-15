#!/bin/sh

echo hostname = `hostname`
m=model.PaddleHub
d=data.PaddleHub

subtask=sst2

for model in `find $gft_checkpoints/fit_examples/  -name 'best' | egrep 'model.PaddleHub.*glue' | egrep $subtask`
	     do

gft_eval --model P:$model \
	 --data P:glue,sst-2 \
	 --metric H:glue,$subtask \
	 --eqn 'classify: labels ~ sentence' \
	 --split val


done
