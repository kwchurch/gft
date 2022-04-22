#!/bin/sh

echo hostname = `hostname`

for model in SetFit/distilbert-base-uncased__ethos_binary__all-train
	     do
    echo MODEL = $model
gft_predict --model H:$model \
    --data H:ethos,binary \
    --split train \
    --eqn 'classify: label ~ text' 
done



