#!/bin/sh

task=stsb
dhub=HuggingFace
hub=PaddleHub
m=model.$hub
d=data.$dhub
model=$hub:$gft_checkpoints/$m/language/$d/glue/$task/ckpt/best

gft_dataset --data $dhub:glue,$task --eqn 'regress: label ~ sentence1 + sentence2' --split test |
gft_predict --model $model

