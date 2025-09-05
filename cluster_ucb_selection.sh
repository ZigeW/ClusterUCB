#!/bin/bash

task=$1
percentage=$2
k=$3
csp=$4
budget=$5
seed=$6

train_file_names=("flan_v2" "cot" "dolly" "oasst1" "code_alpaca" "gpt4_alpaca" "sharegpt" "gsm_train")
train_file_path="./data/{}_data.jsonl"
train_files=("./data/flan_v2_data.jsonl"
             "./data/cot_data.jsonl"
             "./data/dolly_data.jsonl"
             "./data/oasst1_data.jsonl"
             "./data/code_alpaca_data.jsonl"
             "./data/gpt4_alpaca_data.jsonl"
             "./data/sharegpt_data.jsonl"
             "./data/gsm_train_data.jsonl"
            )
ckpt=(20 159 318 477 636)


job_name="llama-2-7b-ucb-beta-p${percentage}-0s-s${seed}"
pretrained_model="./model/llama2_7b"

## ------------ Step 1: Clustering --------------- ##
# 20-step warmup training
source ./train/scripts/warmup_train.sh "${job_name}/k${k}_csp${csp}_bgt${budget}/${task}"
mv "./results/${job_name}/k${k}_csp${csp}_bgt${budget}/${task}/train.log" "./results/${job_name}/k${k}_csp${csp}_bgt${budget}/${task}/checkpoint-${ckpt[epoch]}/train.log"

# calculate training gradient for clustering
model_path="./results/${job_name}/k${k}_csp${csp}_bgt${budget}/${task}/checkpoint-20"
for gpu_id in 0 1 2 3 4 5 6 7
  do
    train_file=${train_file_names[gpu_id]}
    data_path=${train_files[gpu_id]}
    output_path="./grads/${job_name}/k${k}_csp${csp}_bgt${budget}/${task}/ckpt20/${train_file}"
    if [[ ! -d $output_path ]]; then
        mkdir -p $output_path
    fi
    source ./data_selection/scripts/get_train_lora_grads.sh $data_path $model_path $output_path $gpu_id "adam" 2>&1 | tee $output_path/get_gradient.log &
  done
wait

# clustering
cluster_file="./analysis/cluster_${k}_ckpt20_sd${seed}.pt"
if [[ ! -f $cluster_file ]]; then
  export CUDA_VISIBLE_DEVICES=""
  python -m data_selection.kmeans_clustering \
    --train_file_names "${train_file_names[@]}" \
    --train_file_path "${train_file_path}" \
    --grad_path "./grads/${job_name}/k${k}_csp${csp}_bgt${budget}/${task}/ckpt20/{}/dim8192/all_orig.pt" \
    --output_path ${cluster_file} \
    --k ${k} \
    --iters 20 \
    --seed ${seed}
fi

## -------------- Step 2: Inter-cluster data selection & Step 3: Selecting data subset --------------- ##
# select data for the first epoch using computed gradients
# calculate validation gradient
model_path="./results/${job_name}/k${k}_csp${csp}_bgt${budget}/${task}/checkpoint-20"
output_path="./grads/${job_name}/k${k}_csp${csp}_bgt${budget}/${task}/ckpt20/${task}"
if [[ ! -d $output_path ]]; then
    mkdir -p $output_path
fi
source ./data_selection/scripts/get_eval_lora_grads.sh $task "./data/less" $model_path $output_path 7 2>&1 | tee $output_path/get_gradient.log

# matching
gradient_path="./grads/${job_name}/k${k}_csp${csp}_bgt${budget}/${task}/ckpt{}/{}/dim8192/all_orig.pt"
output_path="./scores/${job_name}/k${k}_csp${csp}_bgt${budget}/ckpt20"
if [[ ! -d $output_path ]]; then
  mkdir -p $output_path
fi
ckpt_list=(20)
ckpt_weights=(1.0)
python3 -m data_selection.matching \
        --gradient_path "$gradient_path" \
        --train_file_names "${train_file_names[@]}" \
        --ckpts "${ckpt_list[@]}" \
        --checkpoint_weights "${ckpt_weights[@]}" \
        --validation_gradient_path "$gradient_path" \
        --target_task_names "${task}" \
        --output_path "$output_path"

# select data
output_path="./ucb_selected_data/k${k}_csp${csp}_bgt${budget}/${job_name}/ckpt20"
score_path="./scores/${job_name}/k${k}_csp${csp}_bgt${budget}/ckpt20"
python -m data_selection.write_selected_data \
          --train_file_names "${train_file_names[@]}" \
          --train_files "${train_files[@]}" \
          --target_task_names "${task}" \
          --output_path "$output_path" \
          --score_path "$score_path" \
          --percentage $percentage

# ClusterUCB selection
for epoch in 1 2 3 4
  do
    # train model for one epoch
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    pre_epoch=$((epoch-1))
    if [[ $epoch == 1 ]]; then
      resume_ckpt="none"
    else
      resume_ckpt="./results/${job_name}/k${k}_csp${csp}_bgt${budget}/${task}/checkpoint-${ckpt[pre_epoch]}"
    fi
    data_path="./ucb_selected_data/${job_name}/k${k}_csp${csp}_bgt${budget}/ckpt${ckpt[pre_epoch]}/${task}/top_p${percentage}.jsonl"
    source ./train/scripts/dynamic_train.sh "${job_name}/k${k}_csp${csp}_bgt${budget}/${task}" $resume_ckpt ${data_path}
    mv "./results/${job_name}/k${k}_csp${csp}_bgt${budget}/${task}/train.log" "./results/${job_name}/k${k}_csp${csp}_bgt${budget}/${task}/checkpoint-${ckpt[epoch]}/train.log"

    if [[ $epoch -lt 4 ]]; then
      # calculate validation gradient
      model_path="./results/${job_name}/k${k}_csp${csp}_bgt${budget}/${task}/checkpoint-${ckpt[epoch]}"
      output_path="./grads/${job_name}/k${k}_csp${csp}_bgt${budget}/${task}/ckpt${ckpt[epoch]}/${task}"
      if [[ ! -d $output_path ]]; then
          mkdir -p $output_path
      fi
      source ./data_selection/scripts/get_eval_lora_grads.sh $task "./data/less" $model_path $output_path 7 2>&1 | tee $output_path/get_gradient.log

      # selection using UCB algorithm
      grad_path="./grads/${job_name}/k${k}_csp${csp}_bgt${budget}/{}/ckpt{}/{}/dim8192/all_orig.pt"
      python -m data_selection.cluster_ucb_select_data \
        --task ${task} \
        --train_file_names "${train_file_names[@]}" \
        --train_file_path "${train_file_path}" \
        --model_path ${model_path} \
        --base_model_path ${pretrained_model} \
        --grad_path ${grad_path} \
        --cluster_path ${cluster_file} \
        --output_path "./ucb_selected_data/${job_name}/k${k}_csp${csp}_bgt${budget}/ckpt${ckpt[epoch]}/${task}" \
        --ckpt ${ckpt[epoch]} \
        --method 'ucb_beta' \
        --k ${k} \
        --beta 1 \
        --csp ${csp} \
        --budget ${budget} \
        --top_p ${percentage} \
        --seed ${seed}
    fi
  done

## ------------- Evaluation --------------- ##
model_path="./results/${job_name}/k${k}_csp${csp}_bgt${budget}/${task}/checkpoint-${ckpt[4]}"
if [[ $task == "mmlu" ]]; then
  source ./eval/scripts/eval_mmlu.sh 0 $model_path
elif [[ $task == "gsm" ]]; then
  source ./eval/scripts/eval_gsm.sh 0 $model_path
elif [[ $task == "bbh" ]]; then
  source ./eval/scripts/eval_bbh.sh 0 $model_path
elif [[ $task == "humaneval" ]]; then
  source ./eval/scripts/eval_codex.sh 0 $model_path
elif [[ $task == 'tydiqa' ]]; then
  source ./eval/scripts/eval_tydiqa.sh 0 $model_path
fi

