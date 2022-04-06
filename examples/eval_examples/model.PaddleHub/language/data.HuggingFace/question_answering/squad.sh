#!/bin/sh

echo hostname = `hostname`

b=8

m=model.PaddleHub
d=data.HuggingFace

gft_eval --model P:$gft_checkpoints/fit_examples/$m/language/$d/question_answering/squad/ckpt/best \
    --base_model H:bert-base-cased \
    --data P:squad \
    --metric H:squad \
    --eqn 'classify_spans: answers ~ question + context' \
    -split test



