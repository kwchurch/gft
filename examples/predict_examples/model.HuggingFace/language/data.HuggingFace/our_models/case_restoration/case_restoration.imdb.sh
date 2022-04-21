#!/bin/sh

echo hostname = `hostname`

dir=$1
mkdir -p $dir

python $gft/gft_internals/prepare_datasets/dataset_to_case_labels.py --data H:imdb --splits train,__select_from_train__,test --output $dir/imdb --text_field text

gft_predict --model H:bert-base-uncased \
    --data C:$dir/imdb \
    --output_dir $1 \
    --eqn 'classify_tokens: case_labels ~ tokens' \
    --figure_of_merit overall_accuracy \
    --num_train_epochs 10

