#!/bin/sh

task=rte
hub=PaddleHub
dhub=PaddleHub
m=model.$hub
d=data.$dhub
model=$hub:$gft_checkpoints/$m/language/$d/glue/$task/ckpt/best

gft_predict --data $dhub:glue,$task --eqn 'classify: labels ~ sentence1 + sentence2' --split test --model $model
