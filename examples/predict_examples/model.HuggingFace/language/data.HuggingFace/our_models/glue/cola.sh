#!/bin/sh

task=cola
hub=HuggingFace
m=model.$hub
d=data.$hub
# model=$hub:$gft_checkpoints/$m/language/$d/glue/$task/ckpt/best

# /mnt/big/kwc/morphology/results/20220419/fit_examples/model.HuggingFace/language/data.HuggingFace/hyper/distilbert-base-uncased/glue.lr.2.batch.512/cola/ckpt/best

find $gft_checkpoints/fit_examples/$m/language/$d/hyper/ -name 'best' | egrep $task |
    while read model
	  do
echo model = $model
gft_predict --data $hub:glue,$task --eqn 'classify: label ~ sentence' --split test  --model H:$model
    done




