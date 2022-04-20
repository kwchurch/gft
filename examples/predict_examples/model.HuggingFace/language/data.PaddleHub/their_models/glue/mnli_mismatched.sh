#!/bin/sh

out=$gft_checkpoints/predict_examples/model.HuggingFace/language/data.PaddleHub/their_models/glue/mnli_mismatched
mkdir -p $out

for model in AdapterHub/bert-base-uncased-pf-mnli AdapterHub/roberta-base-pf-mnli textattack/bert-base-uncased-MNLI textattack/roberta-base-MNLI textattack/xlnet-base-cased-MNLI textattack/distilbert-base-uncased-MNLI textattack/facebook-bart-large-MNLI
do
    echo "MODEL $model"
    gft_predict --data P:glue,mnli --eqn 'classify: labels ~ sentence1 + sentence2' --splits train,dev_mismatched,test_matched  --split val --model $model # >$out/predict.out 2>$out/predict.err
done
