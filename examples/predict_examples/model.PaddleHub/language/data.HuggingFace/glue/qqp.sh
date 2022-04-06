#!/bin/sh

task=qqp
dhub=HuggingFace
hub=PaddleHub
m=model.$hub
d=data.$dhub
model=$hub:$gft_checkpoints/$m/language/$d/glue/$task/ckpt/best

gft_predict --data $dhub:glue,$task --eqn 'classify: label ~ question1 + question2' --split test --model $model



