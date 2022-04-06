#!/bin/sh

task=mnli_matched
dhub=HuggingFace
hub=PaddleHub
m=model.$hub
d=data.$dhub
model=$hub:$gft_checkpoints/$m/language/$d/glue/$task/ckpt/best

gft_predict --data $dhub:glue,mnli --eqn 'classify: label ~ premise + hypothesis' --splits train,validation_matched,test_matched  --split test --model $model


