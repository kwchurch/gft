#!/bin/sh

task=stsb

# eduardofv/stsb-m-mt-es-distiluse-base-multilingual-cased-v1 \

for model in cross-encoder/stsb-roberta-large \
cross-encoder/stsb-roberta-base \
cross-encoder/stsb-TinyBERT-L-4 \
cross-encoder/stsb-distilroberta-base \
pritamdeka/S-Biomed-Roberta-snli-multinli-stsb \
howey/bert-base-uncased-stsb \

# for model in AdapterHub/bert-base-uncased-pf-stsb AdapterHub/roberta-base-pf-stsb textattack/bert-base-uncased-STS-B textattack/bert-base-cased-STS-B textattack/distilbert-base-uncased-STS-B textattack/roberta-base-STS-B textattack/albert-base-v2-STS-B textattack/distilbert-base-cased-STS-B textattack/xlnet-base-cased-STS-B textattack/xlnet-large-cased-STS-B
do
    echo "MODEL $model"
    gft_predict --data H:glue,$task --eqn 'regress: label ~ sentence1 + sentence2' --split val --model $model
done
