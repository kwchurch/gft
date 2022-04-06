#!/bin/sh

task=mnli_matched
hub=HuggingFace
dhub=PaddleHub
m=model.$hub
d=data.$dhub
model=$hub:$gft_checkpoints/$m/language/$d/glue/$task/ckpt/best

out=$model

gft_predict --data $dhub:glue,mnli --eqn 'classify: labels ~ sentence1 + sentence2' --splits train,dev_matched,test_matched  --split test --model $model  >$out/predict.out 2>$out/predict.err


