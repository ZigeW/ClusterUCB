#!/bin/bash

source train/scripts/base_training_args.sh

model_path="./model/llama2_7b"
data_seed=42
model_name=$1

train_data=("./data/flan_v2_data.jsonl"
            "./data/cot_data.jsonl"
            "./data/dolly_data.jsonl"
            "./data/oasst1_data.jsonl"
            "./data/code_alpaca_data.jsonl"
            "./data/gpt4_alpaca_data.jsonl"
            "./data/sharegpt_data.jsonl"
            "./data/gsm_train_data.jsonl"
            )

output_dir="./results/${model_name}"
if [[ ! -d $output_dir ]]; then
    mkdir -p $output_dir
fi

# use fsdp for large models
base_training_args="$base_training_args --fsdp 'full_shard auto_wrap' --fsdp_config llama2_7b_finetune"

training_args="$base_training_args \
  --model_name_or_path $model_path \
  --output_dir $output_dir \
  --data_seed $data_seed \
  --save_strategy steps \
  --save_steps 20 \
  --warmup_train True \
  --train_files ${train_data[@]}  2>&1 | tee $output_dir/train.log"

eval "$header" "$training_args"
