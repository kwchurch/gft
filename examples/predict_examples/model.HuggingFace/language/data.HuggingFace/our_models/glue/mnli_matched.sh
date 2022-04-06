#!/bin/sh

task=mnli_matched
hub=HuggingFace
m=model.$hub
d=data.$hub
model=$hub:$gft_checkpoints/$m/language/$d/glue/$task/ckpt/best

gft_predict --data H:glue,mnli --eqn 'classify: label ~ premise + hypothesis' --splits train,validation_matched,test_matched  --split test --model $model


