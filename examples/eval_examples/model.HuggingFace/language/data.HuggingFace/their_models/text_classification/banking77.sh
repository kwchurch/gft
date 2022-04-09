#!/bin/sh

echo hostname = `hostname`

# This model gets an error about batch size (probably because it is for fill-mask task)
# tdopierre/ProtAugment-LM-BANKING77

for model in philschmid/BERT-Banking77 mrm8488/distilroberta-finetuned-banking77 philschmid/DistilBERT-Banking77 philschmid/RoBERTa-Banking77
do
gft_eval --model H:$model \
    --data H:banking77 \
    --eqn 'classify: label ~ text' \
    --split test
done


