#!/bin/sh

task=mnli_matched
dhub=HuggingFace
hub=PaddleHub
m=model.$hub
d=data.$dhub
model=$hub:$gft_checkpoints/$m/language/$d/glue/$task/ckpt/best

    echo "MODEL = $model"
    gft_predict --data H:glue,mnli \
		--eqn 'classify: label ~ premise + hypothesis' \
		--splits train,validation_matched,test_matched \
		--split validation_matched \
		--model H:$model 



