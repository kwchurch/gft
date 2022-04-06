export WANDB_MODE=online
export WANDB_RESUME=allow
export WANDB_PROJECT=wav2vec-finetune
export BATCH_SIZE_PER_GPU=32
export DATA_NAME=timit_asr

MODEL_NAME=facebook/wav2vec2-base
SAVE_PATH=/tmp/output/wav2vec2-base.$DATA_NAME.$BATCH_SIZE_PER_GPU

python $gft/wav2vec2/WER/run_speech_recognition_ctc.py \
	--dataset_name=$DATA_NAME \
	--model_name_or_path=$MODEL_NAME \
	--overwrite_output_dir \
	--output_dir=$SAVE_PATH \
	--train_split_name="train" \
	--num_train_epochs="20" \
	--per_device_train_batch_size=$BATCH_SIZE_PER_GPU \
	--per_device_eval_batch_size="1" \
	--weight_decay="0.005" \
	--learning_rate="1e-4" \
	--warmup_steps="1000" \
	--evaluation_strategy="steps" \
	--text_column_name="text" \
	--save_steps="400" \
	--eval_steps="100" \
	--logging_steps="10" \
	--layerdrop="0.0" \
	--save_total_limit="3" \
	--chars_to_ignore=", ? . ! - \; \: \" “ % ‘ ” �" \
	--fp16 \
	--group_by_length \
	--do_train --do_eval \
	# --push_to_hub \
	# --freeze_feature_extractor \
