#!/bin/sh

task=sst2
dhub=HuggingFace
hub=PaddleHub
m=model.$hub
d=data.$dhub
model=$hub:$gft_checkpoints/$m/language/$d/glue/$task/ckpt/best

gft_dataset --data $dhub:glue,$task --eqn 'classify: label ~ sentence' --split test |
    gft_predict --model $model

