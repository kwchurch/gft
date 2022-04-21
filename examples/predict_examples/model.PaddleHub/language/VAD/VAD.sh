#!/bin/sh

echo hostname = `hostname`

gft_predict --model P:bert-base-cased \
    --data C:$gft/datasets/VAD/VAD \
    --output_dir $1 \
    --eqn 'regress: Valence + Arousal + Dominance ~ Word' \
    --num_train_epochs 10
