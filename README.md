## ClusterUCB

The official repository of **[ClusterUCB: Efficient Gradient-Based Data Selection for Targeted Fine-Tuning of LLMs](https://arxiv.org/abs/2506.10288) [EMNLP 2025 findings]**.

(The codes are modified from [LESS](https://github.com/princeton-nlp/LESS))



### Prerequisites 

- **Installation**: The required Python packages can be installed using:

  ```
  pip install -r requirements.txt
  ```

- **Models**: Please find the pretrained model weights at their official repositories.
- **Instruction tuning data**: Same as *LESS*, we follow the data processing and chatting format of [open-instruct](https://github.com/allenai/open-instruct?tab=readme-ov-file#dataset-preparation). In our experiments, eight datasets are processed: Flan v2, CoT (the chain-of-thought part of Flan v2),  Dolly, Open Assistant v1, GPT4-Alpaca, ShareGPT, GSM8k train split, and Code-Alpaca.
- **Evaluation data**: We use the evaluation tools from [open-instruct](https://github.com/allenai/open-instruct/tree/main/eval). Five benchmarks are included: MMLU, BBH, TydiQA, GSM8k, HumanEval. Please download the evaluation data from their official repositories.



### Data Selection Framework

The proposed **ClusterUCB** framework can be combined with different gradient-based data selection methods. Here, we implement the *Dynamic-ClusterUCB* variant that combines the [dynamic gradient-based data selection (Dynamic)](https://www.sciencedirect.com/science/article/pii/S0950705125011852) with ClusterUCB.

An example of the whole procedure is provided in `cluster_ucb_selectin.sh`. 



Here, we explain it step-by-step.

#### Step 1: Clustering with gradients

The first step of ClusterUCB is to cluster all training data samples according to their gradients.

For Dynamic-ClusterUCB, the gradients used in clustering are the same as those used for the first data selection. 

**Step 1-1:** According to the implementation of Dynamic, a 20-step warmup training should be performed before gradient computation:

```
./train/scripts/warmup_train.sh "$JOB_NAME"
```

**Step 1-2:** Compute the Adam gradients of all training data samples using the model weights obtained from the warmup training:

```
./data_selection/scripts/get_train_lora_grads.sh "$DATA_PATH" "$MODEL_PATH" "$GRAD_OUTPUT_PATH" "$GPU_ID" "adam"
```

**Step 1-3:** Perform k-means clustering. By default, the number of iterations used in k-means is 20. 

The number of clusters can be tuned using the argument "--k".

 [PyKeops](https://github.com/getkeops/keops) are used to accelerate the clustering process.

```
python -m data_selection.kmeans_clustering \
    --train_file_names "${TRAIN_FILE_NAMES[@]}" \
    --train_file_path "$TRAIN_FILE_PATH" \
    --grad_path "$GRAD_PATH"
    --output_path "$CLUSTER_PATH" \
    --k "$K" \
    --iters 20 \
    --seed "$SEED"
```



#### Step 2: Inter-cluster data selection (w/ data subset selection)

After clustering, the inter-cluster data selection using our modified UCB algorithm is performed. The final data subset selection is implemented together with the inter-cluster data selection.

In Dynamic, the data selection is performed multiple times: after one training period, the data subset is re-selected with respect to the current model checkpoint and trained in the next period.

**Step 2-1:** Compute the gradients for the validation data samples using the current model checkpoint. 

The targeted task should be chosen from: "mmlu", "bbh", "tydiqa", "gsm", and "huamneval".

```
./data_selection/scripts/get_eval_lora_grads.sh "$TASK" "$VAL_DATA_PATH" "$MODEL_CKPT" "$GRAD_OUTPUT_PATH" "$GPU_ID" 
```

**Step 2-2:** Perform the UCB algorithm to select the data subset according to the clusters. 

The UCB algorithm can be configured using the argument "--method". The choices are: 'ucb_beta' (our modified UCB algorithm), 'ucb1', 'ucb_threshhard', and 'ucb_threshnormal'. The meaning of the last three methods are explained the Section 4.6 in our paper. 

The cold start ratio can be configured using the argument "--csp".

The computing budget can be configured using the argument "--budget".

The selection percentage can be configured using the argument "--top_p".

```
python -m data_selection.cluster_ucb_select_data \
        --task "$TASK" \
        --train_file_names "${TRAIN_FILE_NAMES[@]}" \
        --train_file_path "$TRAIN_FILE_PATH" \
        --model_path "$MODEL_CKPT" \
        --base_model_path "$PRETRAINED_MODEL_PATH" \
        --grad_path "$GRAD_PATH" \
        --cluster_path "$CLUSTER_PATH" \
        --output_path "$SUBSET_OUTPUT_PATH" \
        --ckpt "CURRENT_EPOCH" \
        --method 'ucb_beta' \
        --k "$K" \
        --beta 1 \
        --csp "$COLD_START_RATIO" \
        --budget "$COMP_BUDGET" \
        --top_p "$SELECT_PERCENT" \
        --seed "$SEED"
```



### Evaluation

We also provide scripts for benchmark evaluation:

```
if [[ $TASK == "mmlu" ]]; then
  source ./eval/scripts/eval_mmlu.sh "$GPU_ID" "$MODEL_CKPT"
elif [[ $TASK == "gsm" ]]; then
  source ./eval/scripts/eval_gsm.sh "$GPU_ID" "$MODEL_CKPT"
elif [[ $TASK == "bbh" ]]; then
  source ./eval/scripts/eval_bbh.sh "$GPU_ID" "$MODEL_CKPT"
elif [[ $TASK == "humaneval" ]]; then
  source ./eval/scripts/eval_codex.sh "$GPU_ID" "$MODEL_CKPT"
elif [[ $TASK == 'tydiqa' ]]; then
  source ./eval/scripts/eval_tydiqa.sh "$GPU_ID" "$MODEL_CKPT"
fi
```

