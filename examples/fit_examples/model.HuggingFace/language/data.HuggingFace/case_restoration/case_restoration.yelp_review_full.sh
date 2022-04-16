#!/bin/sh

echo hostname = `hostname`

dir=$1
mkdir -p $dir

python $gft/gft_internals/prepare_datasets/dataset_to_case_labels.py --data H:yelp_review_full --splits train,__select_from_train__,test --output $dir/yelp_review_full --text_field text

gft_fit --model H:bert-base-uncased \
    --data C:$dir/yelp_review_full \
    --output_dir $1 \
    --eqn 'classify_tokens: case_labels ~ tokens' \
    --figure_of_merit overall_accuracy \
    --per_device_eval_batch_size 16 \
    --per_device_train_batch_size 16 \
    --num_train_epochs 10
