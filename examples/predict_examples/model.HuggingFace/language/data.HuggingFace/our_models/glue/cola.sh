#!/bin/sh

task=cola
hub=HuggingFace
m=model.$hub
d=data.$hub
model=$hub:$gft_checkpoints/$m/language/$d/glue/$task/ckpt/best

gft_predict --data $hub:glue,$task --eqn 'classify: label ~ sentence' --split test  --model $model



