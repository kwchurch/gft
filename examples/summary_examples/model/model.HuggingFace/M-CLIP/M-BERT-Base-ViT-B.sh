#!/bin/sh

# the default task is feature-extraction, but this model will not load with that task
gft_summary --model M-CLIP/M-BERT-Base-ViT-B --task H:default
