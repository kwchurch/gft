#!/bin/sh

echo hostname = `hostname`

# mrm8488/t5-small-finetuned-emotion needs --task text2text-generation
# ditto for mrm8488/t5-base-finetuned-emotion

# AdapterHub/bert-base-uncased-pf-emotion : can't seem to get adapters to work with the most recent version of transformers

for model in bhadresh-savani/distilbert-base-uncased-emotion nateraw/bert-base-uncased-emotion  bhadresh-savani/roberta-base-emotion bhadresh-savani/bert-base-uncased-emotion lewiswatson/distilbert-base-uncased-finetuned-emotion transformersbook/distilbert-base-uncased-finetuned-emotion bhadresh-savani/albert-base-v2-emotion Emanuel/bertweet-emotion-base Emanuel/twitter-emotion-deberta-v3-base reatiny/distilbert-base-uncased-finetuned-emotion Jorgeutd/sagemaker-roberta-base-emotion trnt/twitter_emotions ncduy/bert-base-cased-finetuned-emotion srosy/distilbert-base-uncased-finetuned-emotion MhF/distilbert-base-uncased-finetuned-emotion Crives/distilbert-base-uncased-finetuned-emotion Tommy930/distilbert-base-uncased-finetuned-emotion
	     do
gft_eval --model H:$model \
    --data H:emotion \
    --eqn 'classify: label ~ text' \
    --split test
done



