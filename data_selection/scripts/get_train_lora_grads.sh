#!/bin/bash

train_file=$1 #
model=$2 # path to model
output_path=$3 # path to output
gpu_id=$4
gradient_type=$5

export CUDA_VISIBLE_DEVICES=$gpu_id

if [[ ! -d $output_path ]]; then
    mkdir -p $output_path
fi

python3 -m data_selection.get_info \
--train_file $train_file \
--info_type grads \
--model_path $model \
--base_model_path ./model/llama2_7b \
--output_path $output_path \
--gradient_projection_dimension 8192 \
--gradient_type $gradient_type
