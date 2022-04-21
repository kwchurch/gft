#!/bin/sh

task=cola

out=$gft_checkpoints/predict_examples/model.HuggingFace/language/data.PaddleHub/their_models/glue/$task
mkdir -p $out

for model in textattack/distilbert-base-uncased-CoLA textattack/bert-base-uncased-CoLA textattack/distilbert-base-cased-CoLA textattack/roberta-base-CoLA cointegrated/roberta-large-cola-krishna2020 textattack/albert-base-v2-CoLA gchhablani/bert-base-cased-finetuned-cola yoshitomo-matsubara/bert-base-uncased-cola howey/roberta-large-cola textattack/xlnet-base-cased-CoLA isakbos/Q8BERT_COLA_L_512 EhsanAghazadeh/bert-large-uncased-CoLA_A EhsanAghazadeh/bert-large-uncased-CoLA_B 09panesara/distilbert-base-uncased-finetuned-cola yoshitomo-matsubara/bert-large-uncased-cola  textattack/xlnet-large-cased-CoLA Alireza1044/albert-base-v2-cola pmthangk09/bert-base-uncased-glue-cola kamivao/autonlp-cola_gram-208681 mrm8488/deberta-v3-small-finetuned-cola 123abhiALFLKFO/distilbert-base-uncased-finetuned-cola # Ahren09/distilbert-base-uncased-finetuned-cola

# for model in AdapterHub/bert-base-uncased-pf-cola AdapterHub/roberta-base-pf-cola textattack/bert-base-uncased-CoLA textattack/distilbert-base-uncased-CoLA textattack/distilbert-base-cased-CoLA textattack/roberta-base-CoLA textattack/albert-base-v2-CoLA textattack/xlnet-base-cased-CoLA textattack/facebook-bart-large-CoLA textattack/xlnet-large-cased-CoLA
do
    echo "MODEL $model"
    gft_predict --data P:glue,cola --eqn 'classify: labels ~ sentence' --split val --model $model # >$out/predict.out 2>$out/predict.err
done




