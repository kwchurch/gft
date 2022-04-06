#!/bin/sh

echo hostname = `hostname`

gft_fit --model H:facebook/wav2vec2-base \
    --data H:librispeech_asr,clean \
    --splits train.100,validation,test \
    --output_dir $1 \
    --eqn 'ctc: text ~ audio' \
    --overwrite_output_dir \
    --num_train_epochs 3 \
    --per_device_train_batch_size=4 \
    --gradient_accumulation_steps=8 \
    --learning_rate="3e-4" \
    --warmup_steps="500" \
    --evaluation_strategy="steps" \
    --save_steps="400" \
    --eval_steps="200" \
    --logging_steps="10" \
    --layerdrop="0.0" \
    --save_total_limit="3" \
    --gradient_checkpointing \
    --chars_to_ignore=", ? . ! - \; \: \" “ % ‘ ” �" \
    --fp16 \
    --group_by_length \
    --do_train --do_eval 
