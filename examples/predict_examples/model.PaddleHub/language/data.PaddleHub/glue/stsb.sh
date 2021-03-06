#!/bin/sh

task=stsb
dhub=PaddleHub
hub=PaddleHub
m=model.$hub
d=data.$dhub
model=$hub:$gft_checkpoints/$m/language/$d/glue/$task/ckpt/best

gft_predict --data $dhub:glue,sts-b --eqn 'regress: labels ~ sentence1 + sentence2' --split test --model $model

