#!/bin/sh

task=mnli_mismatched
hub=HuggingFace
dhub=PaddleHub
m=model.$hub
d=data.$dhub
model=$hub:$gft_checkpoints/$m/language/$d/glue/$task/ckpt/best

out=$model

    echo "MODEL = $model"
    gft_predict --data P:glue,mnli \
		--eqn 'classify: labels ~ sentence1 + sentence2' \
		--splits train,dev_mismatched,test_mismatched \
		--split dev_mismatched \
		--model H:$model 

# gft_predict --data $dhub:glue,mnli --eqn 'classify: labels ~ sentence1 + sentence2' --splits train,dev_mismatched,test_matched  --split test --model $model # >$out/predict.out 2>$out/predict.err

