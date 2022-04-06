#!/bin/sh
echo hostname = `hostname`

for model in AdapterHub/bert-base-uncased-pf-mnli AdapterHub/roberta-base-pf-mnli
do
gft_eval --data H:glue,mnli --eqn 'classify: label ~ premise + hypothesis' --splits train,validation_mismatched,test_matched  --split val  --model $model --metric H:glue,mnli
done
