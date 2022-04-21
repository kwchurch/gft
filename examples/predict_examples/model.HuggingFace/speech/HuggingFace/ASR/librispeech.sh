#!/bin/sh

echo hostname = `hostname`

gft_predict --model TBD \
       --base_model H:facebook/wav2vec2-base \
    --data H:librispeech_asr,clean \
    --split test
    --eqn 'ctc: text ~ audio' \
