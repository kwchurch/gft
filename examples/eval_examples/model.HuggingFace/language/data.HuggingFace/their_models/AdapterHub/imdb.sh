#!/bin/sh
echo hostname = `hostname`

for model in AdapterHub/bert-base-uncased-pf-imdb AdapterHub/roberta-base-pf-imdb
	     do
gft_eval --model H:$model \
    --data H:imdb \
    --split test \
    --eqn 'classify: label ~ text' 
done
