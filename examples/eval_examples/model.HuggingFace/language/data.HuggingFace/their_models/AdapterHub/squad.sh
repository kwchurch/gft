#!/bin/sh
echo hostname = `hostname`

# There is a conflict between adapters and classify_spans
# Adapters requires an old version of transformers
# and classify_spans requires a new version of transformers

for model in @ukp/bert-base-uncased_qa_squad1_pfeiffer \
@ukp/roberta-base_qa_squad1_pfeiffer \
@ukp/distilbert-base-uncased_qa_squad1_houlsby \ 
@ukp/distilbert-base-uncased_qa_squad1_pfeiffer \
@ukp/bert-base-uncased_qa_squad1_houlsby \
@ukp/roberta-base_qa_squad1_houlsby \
AdapterHub/bert-base-uncased-pf-squad \
calpt/adapter-bert-base-squad1 \ 
AdapterHub/roberta-base-pf-squad
do

data=squad
# gft_eval --data H:squad --model H:AdapterHub/bert-base-uncased-pf-squad --split test 


b=8

gft_eval --model H:$model \
    --data H:squad \
    --metric H:squad \
    --splits train,validation,validation \
    --split validation \
    --eqn 'classify_spans: answers ~ question + context'

done
