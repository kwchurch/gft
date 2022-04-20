#!/bin/sh

task=mnli_mismatched
hub=HuggingFace
m=model.$hub
d=data.$hub
# model=$hub:$gft_checkpoints/$m/language/$d/glue/$task/ckpt/best

find $gft_checkpoints/fit_examples/$m/language/$d/hyper/ -name 'best' | egrep "$task"_mismatched |
    while read model
	  do
echo model = $model

gft_predict --data H:glue,mnli --eqn 'classify: label ~ premise + hypothesis' --splits train,validation_mismatched,test_matched  --split test --model $model
    done

