#!/bin/sh

task=qnli

out=$gft_checkpoints/predict_examples/model.HuggingFace/language/data.PaddleHub/their_models/glue/$task
mkdir -p $out

for model in AdapterHub/bert-base-uncased-pf-qnli AdapterHub/roberta-base-pf-qnli textattack/bert-base-uncased-QNLI textattack/roberta-base-QNLI textattack/distilbert-base-uncased-QNLI textattack/facebook-bart-large-QNLI textattack/xlnet-base-cased-QNLI
do
    echo "MODEL $model"
    gft_predict --data P:glue,$task --eqn 'classify: labels ~ sentence1 + sentence2' --split val --model $model >$out/predict.out 2>$out/predict.err
done




