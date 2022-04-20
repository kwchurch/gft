#!/bin/sh

task=mrpc
out=$gft_checkpoints/predict_examples/model.HuggingFace/language/data.PaddleHub/their_models/glue/mrpc
mkdir -p $out

for model in textattack/bert-base-uncased-MRPC textattack/roberta-base-MRPC textattack/albert-base-v2-MRPC textattack/distilbert-base-uncased-MRPC textattack/xlnet-large-cased-MRPC textattack/facebook-bart-large-MRPC textattack/distilbert-base-cased-MRPC
do
    echo "MODEL $model"
    gft_predict --data P:glue,mrpc --eqn 'classify: labels ~ sentence1 + sentence2' --split val --model $model # >$out/predict.out 2>$out/predict.err
done

# AdapterHub/bert-base-uncased-pf-mrpc AdapterHub/roberta-base-pf-mrpc
# textattack/xlnet-base-cased-MRPC
