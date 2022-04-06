#!/bin/sh

task=cola

out=$gft_checkpoints/predict_examples/model.HuggingFace/language/data.PaddleHub/their_models/glue/$task
mkdir -p $out

for model in AdapterHub/bert-base-uncased-pf-cola AdapterHub/roberta-base-pf-cola textattack/bert-base-uncased-CoLA textattack/distilbert-base-uncased-CoLA textattack/distilbert-base-cased-CoLA textattack/roberta-base-CoLA textattack/albert-base-v2-CoLA textattack/xlnet-base-cased-CoLA textattack/facebook-bart-large-CoLA textattack/xlnet-large-cased-CoLA
do
    echo "MODEL $model"
    # gft_dataset --data P:glue,cola --eqn 'classify: labels ~ sentence' --split val | gft_predict --model $model
    gft_predict --data P:glue,cola --eqn 'classify: labels ~ sentence' --split val --model $model >$out/predict.out 2>$out/predict.err
done




