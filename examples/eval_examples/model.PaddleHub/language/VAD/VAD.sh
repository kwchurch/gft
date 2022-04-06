#!/bin/sh

echo hostname = `hostname`
m=$model.PaddleHub

gft_eval --model P:$gft_checkpoints/fit_examples/$m/language/VAD/VAD/ckpt/best \
    --base_model H:bert-base-cased \
    --data C:$gft/datasets/VAD/VAD \
    --eqn 'regress: Valence + Arousal + Dominance ~ Word' \
    --split test
