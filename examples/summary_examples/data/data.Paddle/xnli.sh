#!/bin/sh

gft_summary --data P:xnli,en --split dev
gft_summary --data P:xnli,en --split test

for subtask in ar bg de el en es fr hi ru sw th tr ur vi zh
	       do
echo subtask = $subtask
gft_summary --data P:xnli,$subtask --split train
done




