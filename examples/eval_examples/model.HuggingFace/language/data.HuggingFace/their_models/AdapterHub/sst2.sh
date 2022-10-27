#!/bin/sh
echo hostname = `hostname`

task=sst2

for model in AdapterHub/bert-base-uncased-pf-sst2 AdapterHub/roberta-base-pf-sst2
	     do
gft_eval --data H:glue,$task --eqn 'classify: label ~ sentence' --split  val  --model $model --metric H:glue,sst2
done
