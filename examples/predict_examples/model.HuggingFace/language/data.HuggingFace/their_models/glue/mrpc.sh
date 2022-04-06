#!/bin/sh

for model in AdapterHub/bert-base-uncased-pf-mrpc AdapterHub/roberta-base-pf-mrpc textattack/bert-base-uncased-MRPC textattack/roberta-base-MRPC textattack/albert-base-v2-MRPC textattack/distilbert-base-uncased-MRPC textattack/xlnet-large-cased-MRPC textattack/xlnet-base-cased-MRPC textattack/facebook-bart-large-MRPC textattack/distilbert-base-cased-MRPC
do
    echo "MODEL $model"
    gft_predict --data H:glue,mrpc --eqn 'classify: label ~ sentence1 + sentence2' --split test --model $model
done




