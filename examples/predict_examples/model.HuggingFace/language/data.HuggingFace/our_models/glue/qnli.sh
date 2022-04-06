#!/bin/sh

task=qnli
hub=HuggingFace
m=model.$hub
d=data.$hub
model=$hub:$gft_checkpoints/$m/language/$d/glue/$task/ckpt/best

gft_predict --data H:glue,$task --eqn 'classify: label ~ question + sentence' --split test --model $model
