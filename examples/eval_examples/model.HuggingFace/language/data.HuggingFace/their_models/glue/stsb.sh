#!/bin/sh

echo hostname = `hostname`

task=stsb

for model in sentence-transformers/distilbert-base-nli-stsb-mean-tokens sentence-transformers/roberta-base-nli-stsb-mean-tokens sentence-transformers/distilbert-multilingual-nli-stsb-quora-ranking sentence-transformers/stsb-roberta-base sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens sentence-transformers/stsb-bert-base sentence-transformers/stsb-roberta-large cross-encoder/stsb-roberta-large cross-encoder/stsb-roberta-base sentence-transformers/stsb-xlm-r-multilingual sentence-transformers/bert-base-nli-stsb-mean-tokens cross-encoder/stsb-TinyBERT-L-4 sentence-transformers/stsb-distilbert-base sentence-transformers/stsb-mpnet-base-v2 sentence-transformers/stsb-roberta-base-v2 sentence-transformers/roberta-large-nli-stsb-mean-tokens cross-encoder/stsb-distilroberta-base sentence-transformers/xlm-r-bert-base-nli-stsb-mean-tokens Contrastive-Tension/RoBerta-Large-CT-STSb sentence-transformers/stsb-bert-large sentence-transformers/stsb-distilroberta-base-v2 sentence-transformers/bert-large-nli-stsb-mean-tokens sentence-transformers/distilbert-base-nli-stsb-quora-ranking Contrastive-Tension/BERT-Base-Swe-CT-STSb eduardofv/stsb-m-mt-es-distiluse-base-multilingual-cased-v1 mrm8488/distiluse-base-multilingual-cased-v2-finetuned-stsb_multi_mt-es pritamdeka/S-Biomed-Roberta-snli-multinli-stsb howey/bert-base-uncased-stsb Muennighoff/SBERT-base-nli-stsb-v2 AIDA-UPM/mstsb-paraphrase-multilingual-mpnet-base-v2

# for model in AdapterHub/bert-base-uncased-pf-stsb AdapterHub/roberta-base-pf-stsb textattack/bert-base-uncased-STS-B textattack/bert-base-cased-STS-B textattack/distilbert-base-uncased-STS-B textattack/roberta-base-STS-B textattack/albert-base-v2-STS-B textattack/distilbert-base-cased-STS-B textattack/xlnet-base-cased-STS-B textattack/xlnet-large-cased-STS-B
do
    # echo "MODEL $model"
    gft_eval --data H:glue,$task --eqn 'regress: label ~ sentence1 + sentence2' --split val  --model H:$model --metric H:glue,stsb --figure_of_merit pearson 
done
