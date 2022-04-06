#!/bin/sh

task=mnli_mismatched
dhub=PaddleHub
hub=PaddleHub
m=model.$hub
d=data.$dhub
model=$hub:$gft_checkpoints/$m/language/$d/glue/$task/ckpt/best

gft_dataset --data $dhub:glue,mnli --eqn 'classify: labels ~ sentence1 + sentence2' --splits train,dev_mismatched,test_matched  --split test |
gft_predict --model $model

