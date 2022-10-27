#!/bin/sh

for subtask in 'afqmc' 'tnews' 'iflytek' 'ocnli' 'cmnli' 'cluewsc2020' 'csl'
do
    gft_summary --data P:clue,$subtask
done


