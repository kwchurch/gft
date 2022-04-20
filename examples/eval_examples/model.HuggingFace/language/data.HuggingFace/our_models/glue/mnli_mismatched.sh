#!/bin/sh

echo hostname = `hostname`
m=model.HuggingFace
d=data.HuggingFace

for model in `find "$gft_checkpoints"*/fit_examples/  -name 'best' | egrep glue | egrep $subtask`
	     do

gft_eval --model C:$model \
	 --base_model H:__one_of__distilbert-base-uncased,roberta-base,roberta-large,bert-base-cased \
	 --data H:glue,mnli \
	 --metric H:glue,mnli \
	 --eqn 'classify: label ~ premise + hypothesis' \
	 --split validation_mismatched

done
