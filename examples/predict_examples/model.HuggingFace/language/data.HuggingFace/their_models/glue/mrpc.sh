#!/bin/sh

# for model in textattack/bert-base-uncased-MRPC textattack/roberta-base-MRPC textattack/albert-base-v2-MRPC textattack/distilbert-base-uncased-MRPC textattack/xlnet-large-cased-MRPC textattack/xlnet-base-cased-MRPC textattack/facebook-bart-large-MRPC textattack/distilbert-base-cased-MRPC

# textattack/xlnet-base-cased-MRPC does not work; wrong task

for model in textattack/bert-base-uncased-MRPC textattack/roberta-base-MRPC M-FAC/bert-tiny-finetuned-mrpc textattack/albert-base-v2-MRPC ajrae/bert-base-uncased-finetuned-mrpc textattack/distilbert-base-uncased-MRPC howey/roberta-large-mrpc textattack/distilbert-base-cased-MRPC textattack/xlnet-large-cased-MRPC gchhablani/bert-base-cased-finetuned-mrpc ji-xin/roberta_base-MRPC-two_stage shivangi/MRPC_64_128_output ntrnghia/mrpc_vn shivangi/MRPC_output yoshitomo-matsubara/bert-base-uncased-mrpc yoshitomo-matsubara/bert-base-uncased-mrpc_from_bert-large-uncased-mrpc yoshitomo-matsubara/bert-large-uncased-mrpc howey/bert-base-uncased-mrpc Alireza1044/albert-base-v2-mrpc  mrm8488/deberta-v3-small-finetuned-mrpc Maelstrom77/bert-base-uncased-MRPC Maelstrom77/roberta-large-mrpc Riad/finetuned-bert-mrpc M-FAC/bert-mini-finetuned-mrpc NDugar/finetuned-bert-mrpc
do
    echo "MODEL $model"
    gft_predict --data H:glue,mrpc --eqn 'classify: label ~ sentence1 + sentence2' --split test --model $model
done

# AdapterHub/bert-base-uncased-pf-mrpc AdapterHub/roberta-base-pf-mrpc




