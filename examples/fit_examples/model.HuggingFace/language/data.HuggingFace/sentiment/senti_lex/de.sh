#!/bin/sh

echo hostname = `hostname`

# Need to register and download dataset from https://www.kaggle.com/rtatman/sentiment-lexicons-for-81-languages

gft_fit --model H:bert-base-multilingual-uncased \
    --data H:senti_lex,de \
    --data_dir "$HOME/.manual_dirs/senti_lex" \
    --splits train,__select_from_train__,train \
    --output_dir $1 \
    --eqn 'classify: sentiment ~ word' \
    --num_train_epochs 3


