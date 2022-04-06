#!/bin/sh

echo hostname = `hostname`

dir=$1
mkdir -p $dir

python $gft/prepare_datasets/dataset_to_case_labels.py --data H:ag_news --splits train,__select_from_train__,test --output $dir/ag_news --text_field text

gft_fit --model H:bert-base-uncased \
    --data C:$dir/ag_news \
    --output_dir $1 \
    --eqn 'classify_tokens: case_labels ~ tokens' \
    --figure_of_merit overall_accuracy \
    --num_train_epochs 10


