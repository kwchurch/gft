#!/bin/sh

echo hostname = `hostname`

# # ctc training needs dataset dict objects, not numpy arrays
# # --splits train,__select_from_train__,test \

# gft_fit --pretrained H:facebook/wav2vec2-base \
#     --data H:timit_asr \
#     --splits train,test,test \
#     --output_dir $1 \
#     --eqn 'ctc: text ~ audio' \
#     --overwrite_output_dir \
#     --num_train_epochs="20" \
#     --per_device_train_batch_size=32 \
#     --per_device_eval_batch_size="32" \ # changed by kwc
#     --weight_decay="0.005" \
#     --learning_rate="1e-4" \
#     --warmup_steps="1000" \
#     --evaluation_strategy="steps" \
#     --save_steps="400" \
#     --eval_steps="100" \
#     --logging_steps="10" \
#     --layerdrop="0.0" \
#     --save_total_limit="3" \
#     --chars_to_ignore=", ? . ! - \; \: \" “ % ‘ ” �" \
#     --fp16 \
#     --group_by_length \
#     --do_train \
#     --do_eval



#!/bin/sh

# ctc training needs dataset dict objects, not numpy arrays
# --splits train,__select_from_train__,test \

# changed batch size from 32 to 4

gft_fit --model H:facebook/wav2vec2-base \
    --data H:timit_asr \
    --splits train,test,test \
    --output_dir $1 \
    --eqn 'ctc: text ~ audio' \
    --overwrite_output_dir \
    --num_train_epochs="20" \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size="1" \
    --gradient_accumulation_steps=8 \
    --weight_decay="0.005" \
    --learning_rate="1e-4" \
    --warmup_steps="1000" \
    --evaluation_strategy="steps" \
    --save_steps="400" \
    --eval_steps="100" \
    --logging_steps="10" \
    --layerdrop="0.0" \
    --save_total_limit="3" \
    --chars_to_ignore=", ? . ! - \; \: \" “ % ‘ ” �" \
    --fp16 \
    --group_by_length \
    --do_train --do_eval \
