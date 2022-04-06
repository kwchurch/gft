#!/bin/sh

task=mnli_mismatched
dhub=HuggingFace
hub=PaddleHub
m=model.$hub
d=data.$dhub
model=$hub:$gft_checkpoints/$m/language/$d/glue/$task/ckpt/best

gft_dataset --data $dhub:glue,mnli --eqn 'classify: label ~ premise + hypothesis' --splits train,validation_mismatched,test_matched  --split test |
gft_predict --model $model

