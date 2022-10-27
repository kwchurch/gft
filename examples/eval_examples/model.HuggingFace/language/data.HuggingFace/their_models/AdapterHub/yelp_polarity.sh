#!/bin/sh
echo hostname = `hostname`

for model in AdapterHub/bert-base-uncased-pf-yelp_polarity AdapterHub/roberta-base-pf-yelp_polarity
	     do
data=yelp_polarity
gft_eval --data H:yelp_polarity --model H:$model --split test  --eqn 'classify: label ~ text'
done
