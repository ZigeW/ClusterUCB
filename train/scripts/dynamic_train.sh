#!/bin/bash

source train/scripts/base_training_args.sh

model_path="./model/llama2_7b"
data_seed=42
model_name=$1
resume_ckpt=$2
train_data=$3

output_dir="./results/${model_name}"
if [[ ! -d $output_dir ]]; then
    mkdir -p $output_dir
fi

# use fsdp for large models
base_training_args="$base_training_args --fsdp 'full_shard auto_wrap' --fsdp_config llama2_7b_finetune"

if [[ $resume_ckpt == "none" ]]; then
  training_args="$base_training_args \
  --model_name_or_path $model_path \
  --output_dir $output_dir \
  --data_seed $data_seed \
  --train_files "${train_data[@]}" \
  --save_strategy epoch \
  --dynamic_train True 2>&1 | tee $output_dir/train.log"
else
  training_args="$base_training_args \
  --model_name_or_path $model_path \
  --output_dir $output_dir \
  --data_seed $data_seed \
  --train_files "${train_data[@]}" \
  --resume_from_checkpoint ${resume_ckpt}  \
  --save_strategy epoch \
  --dynamic_train True 2>&1 | tee $output_dir/train.log"
fi

eval "$header" "$training_args"
