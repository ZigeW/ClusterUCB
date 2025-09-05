#!/bin/bash

export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=$1

model_name=$2
model_path="$model_name/full_model"
if [[ ! -d $model_path ]]; then
    python ./eval/merge_peft_adapters.py --base_model_name_or_path ../model/llama2_7b --peft_model_path $model_name
fi

output_path="$model_name/eval_bbh"

python -m eval.bbh.run_eval \
--data_dir ../data/bbh \
--save_dir $output_path \
--model $model_path \
--tokenizer $model_path \
--eval_batch_size 1 \
--use_chat_format \
--chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format \
--use_vllm

rm -r $model_path
