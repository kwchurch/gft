#!/bin/sh

task=sst2
dhub=PaddleHub
hub=PaddleHub
m=model.$hub
d=data.$dhub
model=$hub:$gft_checkpoints/$m/language/$d/glue/$task/ckpt/best

gft_dataset --data $dhub:glue,sst-2 --eqn 'classify: labels ~ sentence' --split test |
    gft_predict --model $model

