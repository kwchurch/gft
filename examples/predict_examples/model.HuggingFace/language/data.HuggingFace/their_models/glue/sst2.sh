#!/bin/sh

task=sst2

for model in AdapterHub/bert-base-uncased-pf-sst2 AdapterHub/roberta-base-pf-sst2 textattack/bert-base-uncased-SST-2 textattack/roberta-base-SST-2 textattack/distilbert-base-cased-SST-2 textattack/xlnet-base-cased-SST-2 textattack/albert-base-v2-SST-2 textattack/distilbert-base-uncased-SST-2 textattack/facebook-bart-large-SST-2 textattack/xlnet-large-cased-SST-2
do
    echo "MODEL $model"
    gft_predict --data H:glue,$task --eqn 'classify: label ~ sentence' --split  val --model $model
done


