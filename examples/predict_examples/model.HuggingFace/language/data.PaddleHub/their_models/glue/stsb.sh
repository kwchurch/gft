#!/bin/sh

task=stsb

out=$gft_checkpoints/predict_examples/model.HuggingFace/language/data.PaddleHub/their_models/glue/$task
mkdir -p $out

# textattack/xlnet-base-cased-STS-B does not work; wrong task

for model in AdapterHub/bert-base-uncased-pf-stsb AdapterHub/roberta-base-pf-stsb textattack/bert-base-uncased-STS-B textattack/bert-base-cased-STS-B textattack/distilbert-base-uncased-STS-B textattack/roberta-base-STS-B textattack/albert-base-v2-STS-B textattack/distilbert-base-cased-STS-B textattack/xlnet-large-cased-STS-B
do
    echo "MODEL $model"
    gft_predict --data P:glue,sts-b --eqn 'regress: labels ~ sentence1 + sentence2' --split val  --model $model # >$out/predict.out 2>$out/predict.err
done
