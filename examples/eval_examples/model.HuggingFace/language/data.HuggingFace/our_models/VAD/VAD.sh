#!/bin/sh

echo hostname = `hostname`
m=model.HuggingFace

gft_eval --model H:$gft_checkpoints/fit_examples/$m/language/VAD/VAD/ckpt/best \
	--data C:$gft/datasets/VAD/VAD \
	--eqn 'regress: Valence + Arousal + Dominance ~ Word' \
	--split test

