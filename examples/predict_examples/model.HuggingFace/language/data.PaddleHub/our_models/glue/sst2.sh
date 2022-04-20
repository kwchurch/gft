#!/bin/sh

task=sst2
hub=HuggingFace
dhub=PaddleHub
m=model.$hub
d=data.$dhub
model=$hub:$gft_checkpoints/$m/language/$d/glue/$task/ckpt/best

out=$model

gft_predict --data $dhub:glue,sst-2 --eqn 'classify: labels ~ sentence' --split test --model $model # >$out/predict.out 2>$out/predict.err

