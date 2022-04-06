#!/bin/sh

echo hostname = `hostname`

for model in textattack/bert-base-uncased-MRPC textattack/roberta-base-MRPC M-FAC/bert-tiny-finetuned-mrpc textattack/albert-base-v2-MRPC ajrae/bert-base-uncased-finetuned-mrpc textattack/distilbert-base-uncased-MRPC howey/roberta-large-mrpc textattack/distilbert-base-cased-MRPC textattack/xlnet-large-cased-MRPC gchhablani/bert-base-cased-finetuned-mrpc ji-xin/roberta_base-MRPC-two_stage shivangi/MRPC_64_128_output ntrnghia/mrpc_vn shivangi/MRPC_output yoshitomo-matsubara/bert-base-uncased-mrpc yoshitomo-matsubara/bert-base-uncased-mrpc_from_bert-large-uncased-mrpc yoshitomo-matsubara/bert-large-uncased-mrpc howey/bert-base-uncased-mrpc Alireza1044/albert-base-v2-mrpc AdapterHub/bert-base-uncased-pf-mrpc mrm8488/deberta-v3-small-finetuned-mrpc textattack/xlnet-base-cased-MRPC Maelstrom77/bert-base-uncased-MRPC Maelstrom77/roberta-large-mrpc Riad/finetuned-bert-mrpc M-FAC/bert-mini-finetuned-mrpc NDugar/finetuned-bert-mrpc
do
    gft_eval --data H:glue,mrpc --eqn 'classify: label ~ sentence1 + sentence2' --split val  --model H:$model --metric H:glue,mrpc
done

# bert-base-cased-finetuned-mrpc  does not work: this is actually a fill-mask task, not classify
# textattack/facebook-bart-large-MRPC does not work: there may be a problem in my_auto_model_hf.py

