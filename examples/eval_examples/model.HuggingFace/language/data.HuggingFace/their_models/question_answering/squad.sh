#!/bin/sh

echo hostname = `hostname`

b=8


# distilbert-base-cased-distilled-squad
# bert-large-uncased-whole-word-masking-finetuned-squad
# distilbert-base-uncased-distilled-squad
# bert-large-cased-whole-word-masking-finetuned-squad
# thatdramebaazguy/roberta-base-squad
# dmis-lab/biobert-large-cased-v1.1-squad
# sshleifer/tiny-distilbert-base-cased-distilled-squad
# csarron/bert-base-uncased-squad-v1

for model in distilbert-base-cased-distilled-squad \
bert-large-uncased-whole-word-masking-finetuned-squad \
distilbert-base-uncased-distilled-squad \
bert-large-cased-whole-word-masking-finetuned-squad \
thatdramebaazguy/roberta-base-squad \
dmis-lab/biobert-large-cased-v1.1-squad \
sshleifer/tiny-distilbert-base-cased-distilled-squad \
csarron/bert-base-uncased-squad-v1

	     do

gft_eval --model H:$model \
    --data H:squad \
    --metric H:squad \
    --eqn 'classify_spans: answers ~ question + context' \
    --split val \
    --per_device_train_batch_size 12 \
    --learning_rate 3e-5 \
    --max_seq_length 384 \
    --doc_stride 128

#    --do_not_catch_errors

# gft_eval --model H:$model \
#     --data H:squad \
#     --metric H:squad \
#     --split val \
#     --eqn 'classify_spans: answers ~ question + context' \
#     --debug
done
