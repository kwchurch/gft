#!/bin/sh

echo hostname = `hostname`

for model in cmarkea/distilcamembert-base-sentiment claudio75/xlm-roberta-base-finetuned-marc lewtun/xlm-roberta-base-finetuned-marc philschmid/distilbert-base-multilingual-cased-sentiment lewtun/xlm-roberta-base-finetuned-marc-en Giannipinelli/xlm-roberta-base-finetuned-marc-en begar/xlm-roberta-base-finetuned-marc lewtun/roberta-base-bne-finetuned-amazon_reviews_multi IsabellaKarabasz/roberta-base-bne-finetuned-amazon_reviews_multi philschmid/distilbert-base-multilingual-cased-sentiment-2
	     do
gft_eval --model H:$model \
    --data H:amazon_reviews_multi,all_languages \
    --eqn 'classify: stars ~ review_title + review_body' \
    --split test
done


