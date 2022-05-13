#!/bin/sh

task=qqp

for model in AdapterHub/bert-base-uncased-pf-qqp AdapterHub/roberta-base-pf-qqp textattack/bert-base-uncased-QQP textattack/distilbert-base-cased-QQP textattack/xlnet-base-cased-QQP textattack/albert-base-v2-QQP textattack/xlnet-large-cased-QQP textattack/distilbert-base-uncased-QQP
do
    echo "MODEL $model"
    gft_predict --data H:glue,$task --eqn 'classify: label ~ question1 + question2' --split val --model H:$model
done
