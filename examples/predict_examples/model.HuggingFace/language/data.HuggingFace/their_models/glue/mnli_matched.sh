#!/bin/sh

task=glue
subtask=mnli_matched

for model in AdapterHub/bert-base-uncased-pf-mnli AdapterHub/roberta-base-pf-mnli textattack/bert-base-uncased-MNLI textattack/roberta-base-MNLI textattack/xlnet-base-cased-MNLI textattack/distilbert-base-uncased-MNLI textattack/facebook-bart-large-MNLI
do
    echo "MODEL $model"
    gft_predict --data H:glue,mnli --eqn 'classify: label ~ premise + hypothesis' --splits train,validation_matched,test_matched  --split val  --model $model
done
