#!/bin/sh

task=glue
subtask=mnli_matched

# AdapterHub/bert-base-uncased-pf-mnli AdapterHub/roberta-base-pf-mnli

for model in  textattack/bert-base-uncased-MNLI textattack/roberta-base-MNLI textattack/xlnet-base-cased-MNLI textattack/distilbert-base-uncased-MNLI textattack/facebook-bart-large-MNLI
do

    echo "MODEL = $model"
    gft_predict --data H:glue,mnli \
		--eqn 'classify: label ~ premise + hypothesis' \
		--splits train,validation_mismatched,test_mismatched \
		--split validation_mismatched \
		--model H:$model 

done
