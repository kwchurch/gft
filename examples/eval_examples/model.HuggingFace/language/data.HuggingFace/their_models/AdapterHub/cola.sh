#!/bin/sh
echo hostname = `hostname`

for model in AdapterHub/bert-base-uncased-pf-cola AdapterHub/roberta-base-pf-cola
	     do
data=cola
gft_eval --data H:glue,cola --model H:AdapterHub/bert-base-uncased-pf-cola --split val --eqn 'classify: label ~ sentence' --metric H:glue,cola --figure_of_merit matthews_correlation

done
