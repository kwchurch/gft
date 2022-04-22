#!/bin/sh

echo hostname = `hostname`

task=qqp

for model in textattack/bert-base-uncased-QQP assemblyai/distilbert-base-uncased-qqp yoshitomo-matsubara/bert-base-uncased-qqp howey/roberta-large-qqp gchhablani/bert-base-cased-finetuned-qqp A-bhimany-u08/bert-base-cased-qqp textattack/distilbert-base-cased-QQP textattack/albert-base-v2-QQP  echarlaix/bert-base-uncased-qqp-f87.8-d36-hybrid textattack/distilbert-base-uncased-QQP textattack/xlnet-base-cased-QQP howey/bert-base-uncased-qqp textattack/xlnet-large-cased-QQP yoshitomo-matsubara/bert-base-uncased-qqp_from_bert-large-uncased-qqp Maelstrom77/roberta-large-qqp yoshitomo-matsubara/bert-large-uncased-qqp Alireza1044/albert-base-v2-qqp AnonARR/qqp-bert Maelstrom77/bert-base-uncased-QQP TehranNLP-org/roberta-base-qqp-2e-5-42 persiannlp/mt5-base-parsinlu-qqp-query-paraphrasing M-FAC/bert-mini-finetuned-qqp M-FAC/bert-tiny-finetuned-qqp TehranNLP-org/bert-base-uncased-qqp-2e-5-42 anirudh21/albert-large-v2-finetuned-qqp AdapterHub/bert-base-uncased-pf-qqp zyl1024/bert-base-cased-finetuned-qqp persiannlp/mt5-small-parsinlu-qqp-query-paraphrasing jkgrad/xlnet-base-cased-qqp

# for model in AdapterHub/bert-base-uncased-pf-qqp AdapterHub/roberta-base-pf-qqp textattack/bert-base-uncased-QQP textattack/distilbert-base-cased-QQP textattack/xlnet-base-cased-QQP textattack/albert-base-v2-QQP textattack/xlnet-large-cased-QQP textattack/distilbert-base-uncased-QQP brandontanzhirong/paraphrasing-tool_t5-finetuned-QQP

do
    # echo "MODEL $model"
    gft_eval --data H:glue,$task --eqn 'classify: label ~ question1 + question2' --split val --model H:$model --metric H:glue,qqp
done
