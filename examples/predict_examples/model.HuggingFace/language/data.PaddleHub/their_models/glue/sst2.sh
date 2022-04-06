#!/bin/sh

task=sst2

out=$gft_checkpoints/predict_examples/model.HuggingFace/language/data.PaddleHub/their_models/glue/$task
mkdir -p $out

for model in AdapterHub/bert-base-uncased-pf-sst2 AdapterHub/roberta-base-pf-sst2 textattack/bert-base-uncased-SST-2 textattack/roberta-base-SST-2 textattack/distilbert-base-cased-SST-2 textattack/xlnet-base-cased-SST-2 textattack/albert-base-v2-SST-2 textattack/distilbert-base-uncased-SST-2 textattack/facebook-bart-large-SST-2 textattack/xlnet-large-cased-SST-2
do
    echo "MODEL $model"
    gft_predict --data P:glue,sst-2 --eqn 'classify: labels ~ sentence' --split val --model $model >$out/predict.out 2>$out/predict.err
done


