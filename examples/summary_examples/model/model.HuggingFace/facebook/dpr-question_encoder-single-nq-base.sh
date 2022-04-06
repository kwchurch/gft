#!/bin/sh

# for some reason, HuggingFace has a bad task for this model: feature-extractor

gft_summary --model facebook/dpr-question_encoder-single-nq-base --task H:default

