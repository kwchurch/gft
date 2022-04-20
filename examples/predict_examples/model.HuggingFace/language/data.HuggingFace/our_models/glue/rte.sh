#!/bin/sh

task=rte
hub=HuggingFace
m=model.$hub
d=data.$hub
# model=$hub:$gft_checkpoints/$m/language/$d/glue/$task/ckpt/best

find $gft_checkpoints/fit_examples/$m/language/$d/hyper/ -name 'best' | egrep $task |
    while read model
	  do
echo model = $model

gft_predict --data H:glue,$task --eqn 'classify: label ~ sentence1 + sentence2' --split test  --model $model
    done
