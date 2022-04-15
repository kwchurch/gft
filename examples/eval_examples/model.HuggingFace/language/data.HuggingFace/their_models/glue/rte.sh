#!/bin/sh

echo hostname = `hostname`

task=rte

for model in textattack/bert-base-uncased-RTE textattack/roberta-base-RTE textattack/albert-base-v2-RTE textattack/distilbert-base-uncased-RTE howey/roberta-large-rte gchhablani/bert-base-cased-finetuned-rte yoshitomo-matsubara/bert-base-uncased-rte yoshitomo-matsubara/bert-base-uncased-rte_from_bert-large-uncased-rte howey/bert-base-uncased-rte Alireza1044/albert-base-v2-rte yoshitomo-matsubara/bert-large-uncased-rte textattack/xlnet-base-cased-RTE gchhablani/bert-large-cased-finetuned-rte anirudh21/albert-large-v2-finetuned-rte 
do
    # echo "MODEL $model"
    gft_eval --data H:glue,$task --eqn 'classify: label ~ sentence1 + sentence2' --split val --model H:$model --metric H:glue,rte
done

# AdapterHub/bert-base-uncased-pf-rte
# textattack/facebook-bart-large-RTE


