#!/bin/sh

echo hostname = `hostname`

subtask=cola
m=model.HuggingFace
d=data.HuggingFace

for model in `find "$gft_checkpoints"*/fit_examples/  -name 'best' | egrep 'glue' | egrep $subtask | egrep $m | egrep $d`
	     do
gft_eval --model C:$model \
	 --base_model H:__one_of__distilbert-base-uncased,roberta-base,roberta-large,bert-base-cased \
	 --data H:glue,$subtask \
	 --metric H:glue,$subtask \
	 --figure_of_merit matthews_correlation \
	 --eqn 'classify: label ~ sentence' \
	 --split val
done
