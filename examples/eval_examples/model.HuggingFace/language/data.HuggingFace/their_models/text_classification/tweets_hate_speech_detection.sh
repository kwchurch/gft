#!/bin/sh

echo hostname = `hostname`

# Narrativa/byt5-base-tweet-hate-detection --task text-generation

for model in mrm8488/distilroberta-finetuned-tweets-hate-speech
	     do
gft_eval --model $model \
    --data H:tweets_hate_speech_detection \
    --eqn 'classify: label ~ tweet' \
    --split train \
    --do_not_catch_errors
done




