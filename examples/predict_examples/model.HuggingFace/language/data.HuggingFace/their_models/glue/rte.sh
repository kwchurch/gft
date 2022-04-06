#!/bin/sh

task=rte

for model in AdapterHub/bert-base-uncased-pf-rte AdapterHub/roberta-base-pf-rte textattack/bert-base-uncased-RTE textattack/roberta-base-RTE textattack/facebook-bart-large-RTE textattack/albert-base-v2-RTE textattack/distilbert-base-uncased-RTE textattack/facebook-bart-base-glue-RTE textattack/facebook-bart-base-RTE textattack/xlnet-base-cased-RTE
do
    echo "MODEL $model"
    gft_predict --data H:glue,$task --eqn 'classify: label ~ sentence1 + sentence2' --split val --model $model
done
