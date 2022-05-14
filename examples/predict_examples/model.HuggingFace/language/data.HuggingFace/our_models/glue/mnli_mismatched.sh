#!/bin/sh

task=mnli_mismatched
hub=HuggingFace
m=model.$hub
d=data.$hub
# model=$hub:$gft_checkpoints/$m/language/$d/glue/$task/ckpt/best

find $gft_checkpoints/fit_examples/$m/language/$d/hyper/ -name 'best' | egrep "$task"_mismatched |
    while read model
	  do

    echo "MODEL = $model"
    gft_predict --data P:glue,mnli \
		--eqn 'classify: labels ~ sentence1 + sentence2' \
		--splits train,dev_mismatched,test_mismatched \
		--split dev_mismatched \
		--model H:$model 

    done

