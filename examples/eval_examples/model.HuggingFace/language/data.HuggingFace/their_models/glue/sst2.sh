#!/bin/sh

echo hostname = `hostname`

task=sst2

for model in philschmid/MiniLM-L6-H384-uncased-sst2 echarlaix/bert-base-uncased-sst2-acc91.1-d37-hybrid howey/bert-base-uncased-sst2 Alireza1044/albert-base-v2-sst2 mfuntowicz/bert-base-cased-finetuned-sst2 gchhablani/fnet-base-finetuned-sst2 yoshitomo-matsubara/bert-large-uncased-sst2 Bhumika/roberta-base-finetuned-sst2 assemblyai/distilbert-base-uncased-sst2 howey/roberta-large-sst2 gchhablani/bert-base-cased-finetuned-sst2 M-FAC/bert-tiny-finetuned-sst2 mrm8488/deberta-v3-small-finetuned-sst2 assemblyai/bert-large-uncased-sst2 neal49/distilbert-sst2-1 EhsanAghazadeh/bert-based-uncased-sst2-e5 yoshitomo-matsubara/bert-base-uncased-sst2 bhadresh-savani/distilbert-base-uncased-sentiment-sst2 pmthangk09/bert-base-uncased-glue-sst2 neal49/distilbert-sst2-freeze moshew/tiny-bert-aug-sst2-distilled yoshitomo-matsubara/bert-base-uncased-sst2_from_bert-large-uncased-sst2 o2poi/sst2-eda-albert philschmid/tiny-bert-sst2-distilled M-FAC/bert-mini-finetuned-sst2 o2poi/sst2-eda-bert ikevin98/bert-base-uncased-finetuned-sst2 o2poi/sst2-eda-bert-uncased

# for model in AdapterHub/bert-base-uncased-pf-sst2 AdapterHub/roberta-base-pf-sst2 textattack/bert-base-uncased-SST-2 textattack/roberta-base-SST-2 textattack/distilbert-base-cased-SST-2 textattack/xlnet-base-cased-SST-2 textattack/albert-base-v2-SST-2 textattack/distilbert-base-uncased-SST-2 textattack/facebook-bart-large-SST-2 textattack/xlnet-large-cased-SST-2 AdapterHub/roberta-base-pf-sst2 philschmid/Infinity_cpu_MiniLM_L6_H384_uncased_sst2
do
    # echo "MODEL $model"
    gft_eval --data H:glue,$task --eqn 'classify: label ~ sentence' --split  val  --model H:$model --metric H:glue,sst2
done


