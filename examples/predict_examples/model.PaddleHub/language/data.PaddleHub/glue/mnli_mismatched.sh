#!/bin/sh

task=mnli_mismatched
dhub=PaddleHub
hub=PaddleHub
m=model.$hub
d=data.$dhub
model=$gft_checkpoints/$m/language/$d/glue/mnli/ckpt/best

    echo "MODEL = $model"
    gft_predict --data P:glue,mnli \
		--eqn 'classify: labels ~ sentence1 + sentence2' \
		--splits train,dev_mismatched,test_mismatched \
		--split dev_mismatched \
		--model P:$model 


