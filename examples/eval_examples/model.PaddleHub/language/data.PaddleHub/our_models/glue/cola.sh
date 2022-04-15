#!/bin/sh

echo hostname = `hostname`

subtask=cola
m=model.PaddleHub
d=data.PaddleHub

for model in `find $gft_checkpoints/fit_examples/  -name 'best' | egrep 'model.PaddleHub.*glue' | egrep $subtask`
	     do
gft_eval --model P:$model \
	 --data P:glue,$subtask \
	 --metric H:glue,$subtask \
	 --figure_of_merit matthews_correlation \
	 --eqn 'classify: labels ~ sentence' \
	 --split val
done
