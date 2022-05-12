#!/bin/sh

# dataset: banking77	model: philschmid/BERT-Banking77	downloads: 8528	likes: 0
# dataset: banking77	model: mrm8488/distilroberta-finetuned-banking77	downloads: 66	likes: 2
# dataset: banking77	model: philschmid/DistilBERT-Banking77	downloads: 56	likes: 0
# dataset: banking77	model: philschmid/RoBERTa-Banking77	downloads: 16	likes: 0

# tdopierre/ProtAugment-LM-BANKING77 does not work because the task is fill-mask

for model in philschmid/BERT-Banking77  mrm8488/distilroberta-finetuned-banking77 philschmid/DistilBERT-Banking77 philschmid/RoBERTa-Banking77
do
echo model = $model
gft_predict --data H:banking77 --model $model --eqn  'classify: label ~ text' --split test
done

