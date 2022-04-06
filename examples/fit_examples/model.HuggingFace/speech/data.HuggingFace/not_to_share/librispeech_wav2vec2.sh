export WANDB_MODE=online
export WANDB_RESUME=allow
export WANDB_PROJECT=wav2vec-finetune
export BATCH_SIZE_PER_GPU=4
export DATA_NAME=librispeech_asr

if [ -z "$1" ]; then
	echo "fine-tune public wav2vec2-base"
	export MODEL_NAME=facebook/wav2vec2-large-lv60
	export SAVE_PATH=output/wav2vec2-large.$DATA_NAME-clean-100h.$BATCH_SIZE_PER_GPU
else
	echo "fine-tune checkpoint"
	export MODEL_NAME=$1
	export SAVE_PATH=output/$2.$DATA_NAME-clean-100h.$BATCH_SIZE_PER_GPU
fi

srun -p P100 --gres=gpu:8 -n 1 --cpus-per-task=40 \
python -m torch.distributed.launch --nproc_per_node 8 $gft/wav2vec2/WER/run_speech_recognition_ctc.py \
	--dataset_name=$DATA_NAME \
	--model_name_or_path=$MODEL_NAME \
	--dataset_config_name="clean" \
	--train_split_name="train.100" \
	--eval_split_name="validation" \
	--preprocessing_num_workers="40" \
	--overwrite_output_dir \
	--output_dir=$SAVE_PATH \
	--num_train_epochs="3" \
	--per_device_train_batch_size=$BATCH_SIZE_PER_GPU \
	--gradient_accumulation_steps="1" \
	--learning_rate="3e-4" \
	--warmup_steps="500" \
	--evaluation_strategy="steps" \
	--text_column_name="text" \
	--save_steps="400" \
	--eval_steps="200" \
	--logging_steps="10" \
	--layerdrop="0.0" \
	--save_total_limit="3" \
	--gradient_checkpointing \
	--chars_to_ignore=", ? . ! - \; \: \" “ % ‘ ” �" \
	--fp16 \
	--group_by_length \
	--do_train --do_eval \
	# --push_to_hub \
	# --freeze_feature_extractor \
