#!/bin/sh

task=glue
subtask=cola

for model in AdapterHub/bert-base-uncased-pf-cola AdapterHub/roberta-base-pf-cola textattack/bert-base-uncased-CoLA textattack/distilbert-base-uncased-CoLA textattack/distilbert-base-cased-CoLA textattack/roberta-base-CoLA textattack/albert-base-v2-CoLA textattack/xlnet-base-cased-CoLA textattack/facebook-bart-large-CoLA textattack/xlnet-large-cased-CoLA
do
    echo "MODEL $model"
    gft_predict --data H:glue,cola --eqn 'classify: label ~ sentence' --split val --model $model
done




