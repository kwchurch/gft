#!/bin/sh
echo hostname = `hostname`

model=AdapterHub/bert-base-uncased-pf-mit_movie_trivia
data=mit_movie_trivia
gft_eval --data H:mit_movie_trivia --model H:AdapterHub/bert-base-uncased-pf-mit_movie_trivia --split test 
