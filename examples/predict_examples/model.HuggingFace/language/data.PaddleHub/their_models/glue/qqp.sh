#!/bin/sh

task=qqp

# out=$gft_checkpoints/predict_examples/model.HuggingFace/language/data.PaddleHub/their_models/glue/$task
# mkdir -p $out

for model in AdapterHub/bert-base-uncased-pf-qqp AdapterHub/roberta-base-pf-qqp textattack/bert-base-uncased-QQP textattack/distilbert-base-cased-QQP textattack/xlnet-base-cased-QQP textattack/albert-base-v2-QQP textattack/xlnet-large-cased-QQP textattack/distilbert-base-uncased-QQP
do
    echo "MODEL $model"
    gft_predict --data P:glue,$task --eqn 'classify: labels ~ sentence1 + sentence2' --split val --model H:$model # >$out/predict.out 2>$out/predict.err
done
