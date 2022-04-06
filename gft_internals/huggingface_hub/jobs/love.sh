#!/bin/sh

lab=$gft/huggingface_hub/huggingface_models_with_labels.txt
models=`awk 'NF < 20' $lab | egrep -i "$1"  | awk '{print rand() "\t" $0}' | sort | cut -f2- | sed 6q | cut -f1`
for model in $models
     do
     out=`echo 'I love you.' | gft_predict --task text-classification --model $model`
     echo $model $out
     done

