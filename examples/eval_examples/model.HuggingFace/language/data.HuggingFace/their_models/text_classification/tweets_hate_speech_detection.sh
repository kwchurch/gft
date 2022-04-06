#!/bin/sh

echo hostname = `hostname`

for model in Narrativa/byt5-base-tweet-hate-detection mrm8488/distilroberta-finetuned-tweets-hate-speech
	     do
gft_eval --model $model \
    --data H:tweets_hate_speech_detection \
    --eqn 'classify: label ~ tweet' \
    --split train
done




