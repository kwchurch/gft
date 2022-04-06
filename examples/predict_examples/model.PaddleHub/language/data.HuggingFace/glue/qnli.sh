#!/bin/sh

task=qnli
dhub=HuggingFace
hub=PaddleHub
m=model.$hub
d=data.$dhub
model=$hub:$gft_checkpoints/$m/language/$d/glue/$task/ckpt/best

gft_dataset --data $dhub:glue,$task --eqn 'classify: label ~ question + sentence' --split test |
gft_predict --model $model
