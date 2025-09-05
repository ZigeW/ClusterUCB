#!/bin/bash

GPUS_PER_NODE=$(python -c 'import torch; print(torch.cuda.device_count())')

ID=$RANDOM
export header="torchrun --nproc_per_node $GPUS_PER_NODE --nnodes 1 \
--rdzv-id=$ID --rdzv_backend c10d \
-m train.train"

export base_training_args="--do_train True \
--max_seq_length 4096 \
--use_fast_tokenizer True \
--lr_scheduler_type linear \
--warmup_ratio 0.03 \
--weight_decay 0.0 \
--evaluation_strategy no \
--logging_steps 1 \
--num_train_epochs 4 \
--bf16 False \
--tf32 False \
--fp16 True \
--overwrite_output_dir False \
--report_to none \
--optim adamw_torch \
--seed 0 \
--lora True \
--lora_r 128 \
--lora_alpha 512 \
--lora_dropout 0.1 \
--lora_target_modules q_proj k_proj v_proj o_proj \
--learning_rate 2e-05 \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 16 \
--gradient_checkpointing True"