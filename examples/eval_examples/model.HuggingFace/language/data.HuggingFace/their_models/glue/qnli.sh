#!/bin/sh

echo hostname = `hostname`

task=qnli

for model in cross-encoder/qnli-electra-base textattack/bert-base-uncased-QNLI textattack/roberta-base-QNLI cross-encoder/qnli-distilroberta-base howey/roberta-large-qnli howey/electra-large-qnli textattack/distilbert-base-uncased-QNLI gchhablani/bert-base-cased-finetuned-qnli mervenoyan/PubMedBERT-QNLI savasy/bert-turkish-uncased-qnli yoshitomo-matsubara/bert-base-uncased-qnli yoshitomo-matsubara/bert-base-uncased-qnli_from_bert-large-uncased-qnli howey/bert-base-uncased-qnli textattack/xlnet-base-cased-QNLI yoshitomo-matsubara/bert-large-uncased-qnli  Alireza1044/albert-base-v2-qnli anirudh21/albert-base-v2-finetuned-qnli M-FAC/bert-tiny-finetuned-qnli Li/bert-base-uncased-qnli bioformers/bioformer-cased-v1.0-qnli mrm8488/deberta-v3-small-finetuned-qnli M-FAC/bert-mini-finetuned-qnli anirudh21/bert-base-uncased-finetuned-qnli anirudh21/albert-large-v2-finetuned-qnli anirudh21/distilbert-base-uncased-finetuned-qnli navteca/qnli-electra-base
# for model in AdapterHub/bert-base-uncased-pf-qnli AdapterHub/roberta-base-pf-qnli textattack/bert-base-uncased-QNLI textattack/roberta-base-QNLI textattack/distilbert-base-uncased-QNLI textattack/facebook-bart-large-QNLI textattack/xlnet-base-cased-QNLI AdapterHub/bert-base-uncased-pf-qnli textattack/facebook-bart-large-QNLI mrm8488/bert-uncased-finetuned-qnli 
do
    # echo "MODEL $model"
    gft_eval --data H:glue,$task --eqn 'classify: label ~ question + sentence' --split val  --model H:$model --metric H:glue,qnli
done




