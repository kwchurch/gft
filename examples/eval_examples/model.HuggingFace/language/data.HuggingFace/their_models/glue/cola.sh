#!/bin/sh

echo hostname = `hostname`

task=glue
subtask=cola

for model in textattack/distilbert-base-uncased-CoLA textattack/bert-base-uncased-CoLA textattack/distilbert-base-cased-CoLA textattack/roberta-base-CoLA cointegrated/roberta-large-cola-krishna2020 textattack/albert-base-v2-CoLA gchhablani/bert-base-cased-finetuned-cola yoshitomo-matsubara/bert-base-uncased-cola howey/roberta-large-cola textattack/xlnet-base-cased-CoLA isakbos/Q8BERT_COLA_L_512 EhsanAghazadeh/bert-large-uncased-CoLA_A EhsanAghazadeh/bert-large-uncased-CoLA_B 09panesara/distilbert-base-uncased-finetuned-cola yoshitomo-matsubara/bert-large-uncased-cola  textattack/xlnet-large-cased-CoLA Alireza1044/albert-base-v2-cola pmthangk09/bert-base-uncased-glue-cola kamivao/autonlp-cola_gram-208681 mrm8488/deberta-v3-small-finetuned-cola 123abhiALFLKFO/distilbert-base-uncased-finetuned-cola Ahren09/distilbert-base-uncased-finetuned-cola

do
    gft_eval --data H:glue,cola --eqn 'classify: label ~ sentence' --split val  --model H:$model --metric H:glue,cola --figure_of_merit matthews_correlation
done

# textattack/facebook-bart-large-CoLA does not work
# This is from https://huggingface.co/textattack/facebook-bart-large-CoLA, but does not work
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# tokenizer = AutoTokenizer.from_pretrained("textattack/facebook-bart-large-CoLA")
# model = AutoModelForSequenceClassification.from_pretrained("textattack/facebook-bart-large-CoLA")


# Ahren09/distilbert-base-uncased-finetuned-cola does not work
# TypeError: forward() got an unexpected keyword argument 'token_type_ids'

