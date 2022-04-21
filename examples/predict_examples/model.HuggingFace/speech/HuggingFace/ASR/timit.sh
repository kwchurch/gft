#!/bin/sh

echo hostname = `hostname`

gft_predict --model TBD \
	    --base_model H:facebook/wav2vec2-base \
    --data H:timit_asr \
    --eqn 'ctc: text ~ audio' \
    --split test

