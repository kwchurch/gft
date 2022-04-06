#!/bin/sh

echo hostname = `hostname`

task=glue
subtask=mnli_matched

for model in facebook/bart-large-mnli roberta-large-mnli typeform/distilbert-base-uncased-mnli MoritzLaurer/mDeBERTa-v3-base-mnli-xnli valhalla/distilbart-mnli-12-1 Narsil/deberta-large-mnli-zero-cls valhalla/distilbart-mnli-12-6 textattack/roberta-base-MNLI microsoft/deberta-large-mnli microsoft/deberta-xlarge-mnli ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli valhalla/distilbart-mnli-12-3 joeddav/bart-large-mnli-yahoo-answers typeform/mobilebert-uncased-mnli textattack/bert-base-uncased-MNLI valhalla/distilbart-mnli-12-9 MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli prajjwal1/bert-medium-mnli microsoft/deberta-base-mnli NbAiLab/nb-bert-base-mnli MoritzLaurer/xtremedistil-l6-h256-mnli-fever-anli-ling-binary alan-turing-institute/mt5-large-finetuned-mnli-xtreme-xnli prajjwal1/albert-base-v2-mnli symanto/sn-xlm-roberta-base-snli-mnli-anli-xnli textattack/xlnet-base-cased-MNLI typeform/roberta-large-mnli microsoft/deberta-v2-xlarge-mnli ynie/albert-xxlarge-v2-snli_mnli_fever_anli_R1_R2_R3-nli microsoft/deberta-v2-xxlarge-mnli prajjwal1/bert-mini-mnli
# for model in AdapterHub/bert-base-uncased-pf-mnli AdapterHub/roberta-base-pf-mnli textattack/bert-base-uncased-MNLI textattack/roberta-base-MNLI textattack/xlnet-base-cased-MNLI textattack/distilbert-base-uncased-MNLI textattack/facebook-bart-large-MNLI
do
    # echo "MODEL $model"
    gft_eval --data H:glue,mnli --eqn 'classify: label ~ premise + hypothesis' --splits train,validation_matched,test_matched  --split val  --model H:$model --metric H:glue,mnli
done
