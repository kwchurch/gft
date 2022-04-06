#!/bin/sh

task=qnli

for model in AdapterHub/bert-base-uncased-pf-qnli AdapterHub/roberta-base-pf-qnli textattack/bert-base-uncased-QNLI textattack/roberta-base-QNLI textattack/distilbert-base-uncased-QNLI textattack/facebook-bart-large-QNLI textattack/xlnet-base-cased-QNLI
do
    echo "MODEL $model"
    gft_predict --data H:glue,$task --eqn 'classify: label ~ question + sentence' --split val --model $model
done




