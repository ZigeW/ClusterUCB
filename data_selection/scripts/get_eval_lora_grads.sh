#!/bin/bash

# for validation data, we should always get gradients with sgd
task=$1 # tydiqa, mmlu
data_dir=$2 # path to data
model=$3 # path to model
output_path=$4 # path to output
dims=8192 # dimension of projection, can be a list

export CUDA_VISIBLE_DEVICES=$5

if [[ ! -d $output_path ]]; then
    mkdir -p $output_path
fi

python3 -m data_selection.get_info \
--task $task \
--info_type grads \
--model_path $model \
--base_model_path ../model/llama2_7b \
--output_path $output_path \
--gradient_projection_dimension $dims \
--gradient_type sgd \
--data_dir $data_dir \
--torch_dtype float16 \
--max_length 1400
