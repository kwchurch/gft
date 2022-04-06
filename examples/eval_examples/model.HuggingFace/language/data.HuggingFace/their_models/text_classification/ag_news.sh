#!/bin/sh

echo hostname = `hostname`

for model in mrm8488/bert-mini-finetuned-age_news-classification joeddav/distilbert-base-uncased-agnews-student nateraw/bert-base-uncased-ag-news mrm8488/distilroberta-finetuned-age_news-classification andi611/distilbert-base-uncased-ner-agnews lucasresck/bert-base-cased-ag-news fabriceyhc/bert-base-uncased-ag_news
	     do
gft_eval --model H:$model \
    --data H:ag_news \
    --eqn 'classify: label ~ text' \
    --split test
done
