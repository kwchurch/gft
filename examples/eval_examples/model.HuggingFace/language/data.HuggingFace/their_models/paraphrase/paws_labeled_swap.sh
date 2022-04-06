#!/bin/sh

echo hostname = `hostname`

for model in mrm8488/bert-base-german-dbmdz-cased-finetuned-pawsx-de ziqingyang/XLMRobertaBaseForPAWSX-en mrm8488/camembert-base-finetuned-pawsx-fr mrm8488/RuPERTa-base-finetuned-pawsx-es bertin-project/bertin-base-paws-x-es maximedb/paws-x-all-x-en mrm8488/electricidad-base-finetuned-pawsx-es maximedb/paws-x-all mrm8488/electricidad-small-finetuned-paws-x-es
do
gft_eval --model H:$model \
    --data H:paws,labeled_swap \
    --eqn 'classify: label ~ sentence1 + sentence2' \
    --split test
done

