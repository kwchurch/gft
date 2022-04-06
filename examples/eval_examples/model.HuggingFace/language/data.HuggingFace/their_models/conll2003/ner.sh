#!/bin/sh

echo hostname = `hostname`

for model in xlm-roberta-large-finetuned-conll03-english dbmdz/bert-large-cased-finetuned-conll03-english sshleifer/tiny-dbmdz-bert-large-cased-finetuned-conll03-english elastic/distilbert-base-uncased-finetuned-conll03-english xlm-roberta-large-finetuned-conll03-german dbmdz/electra-large-discriminator-finetuned-conll03-english elastic/distilbert-base-cased-finetuned-conll03-english philschmid/distilroberta-base-ner-wikiann-conll2003-3-class philschmid/distilroberta-base-ner-conll2003 bertin-project/bertin-base-ner-conll2002-es wietsedv/bert-base-dutch-cased-finetuned-conll2002-ner
do
gft_eval --model H:$model \
    --data H:conll2003 \
    --eqn 'classify_tokens: ner_tags ~ tokens' \
    --figure_of_merit overall_accuracy \
    --is_split_into_words \
    --split test
done
	     

