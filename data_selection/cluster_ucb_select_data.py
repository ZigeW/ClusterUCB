import json
import os
import numpy as np
import random
import argparse
import torch
from copy import deepcopy
from typing import Any
import tqdm
import math
import bisect
from scipy.stats import Normal
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trak.projectors import BasicProjector, CudaProjector, ProjectionType
from torch.nn.functional import normalize
from data_selection.get_training_dataset import get_training_dataset
from data_selection.collect_grad_reps import (obtain_gradients_with_adam, get_trak_projector, prepare_optimizer_state,
                                              get_number_of_params, prepare_batch)

N_SUBTASKS = {"mmlu": 57, "bbh": 27, "tydiqa": 9, "humaneval": 1, "gsm": 1}


def load_model(model_name_or_path: str,
               base_model_name_or_path: str,
               torch_dtype: Any = torch.bfloat16) -> Any:
    is_peft = os.path.exists(os.path.join(
        model_name_or_path, "adapter_config.json"))
    if is_peft:
        # load this way to make sure that optimizer states match the model structure
        config = LoraConfig.from_pretrained(model_name_or_path)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name_or_path, torch_dtype=torch_dtype, device_map="auto")
        model = PeftModel.from_pretrained(
            base_model, model_name_or_path, device_map="auto")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, torch_dtype=torch_dtype, device_map="auto")

    for name, param in model.named_parameters():
        if 'lora' in name or 'Lora' in name:
            param.requires_grad = True
    return model


def build_data_list(args):
    total_candidates = []
    for i, train_file in enumerate(args.train_file_names):
        with open(args.train_file_path.format(train_file), 'r') as f:
            for j, line in enumerate(f.readlines()):
                data = json.loads(line)
                total_candidates.append(f"{data['dataset']}//{j}")

    return total_candidates


def matching(train_grad, val_grad):
    return torch.matmul(train_grad, val_grad.transpose(0, 1)).reshape(train_grad.shape[0], N_SUBTASKS[args.task], -1).mean(
            -1).max(-1)[0]


def get_training_gradients(dataset, model, optimizer_state, proj):
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    # prepare optimization states
    m, v = prepare_optimizer_state(model, optimizer_state, device)

    full_grads = []
    projected_grads = []
    project_interval = 8
    for i in range(len(dataset)):
        batch = dataset[[i]]
        prepare_batch(batch)
        model.zero_grad()
        try:
            vectorized_grads = obtain_gradients_with_adam(model, batch, m, v)
        except Exception as e:
            print(e)
            number_of_params = get_number_of_params(model, verbose=False)
            vectorized_grads = torch.zeros((number_of_params,), dtype=dtype)

        full_grads.append(vectorized_grads)

        if i % project_interval == 0:
            full_grads = torch.stack(full_grads).to(torch.float16)
            projected_grads.append(proj.project(full_grads, model_id=0).cpu())
            full_grads = []

    if len(full_grads) > 0:
        full_grads = torch.stack(full_grads).to(torch.float16)
        projected_grads.append(proj.project(full_grads, model_id=0).cpu())
        full_grads = []
    projected_grads = torch.cat(projected_grads)
    normalized_grads = normalize(projected_grads, dim=1)
    # print(f"normalized gradient shape: {normalized_grads.shape}")
    return normalized_grads


class UCB1():
    def __init__(self, counts, values):
        self.counts = counts
        self.values = values
        return

    def initialize(self, n_arms):
        self.counts = [0 for col in range(n_arms)]
        self.values = [0.0 for col in range(n_arms)]
        return

    def select_arm(self):
        n_arms = len(self.counts)
        for arm in range(n_arms):
            if self.counts[arm] == 0:
                return arm

        ucb_values = [0.0 for arm in range(n_arms)]
        total_counts = sum(self.counts)

        for arm in range(n_arms):
            bonus = math.sqrt((2 * math.log(total_counts)) / float(self.counts[arm]))
            ucb_values[arm] = self.values[arm] + bonus
        return ucb_values.index(max(ucb_values))

    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] = self.counts[chosen_arm] + 1
        n = self.counts[chosen_arm]

        value = self.values[chosen_arm]
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.values[chosen_arm] = new_value
        return

    def update_batch(self, chosen_arm, reward):
        for i in range(len(list(reward))):
            self.counts[chosen_arm] = self.counts[chosen_arm] + 1
            n = self.counts[chosen_arm]

            value = self.values[chosen_arm]
            new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward[i]
            self.values[chosen_arm] = new_value
        return


class UCB_Random():
    def __init__(self, n_arms, draw_limits):
        self.counts = [0 for col in range(n_arms)]
        self.valid = np.ones(n_arms, dtype=np.int32)
        self.arms = np.arange(n_arms)
        self.draw_limits = draw_limits
        return

    def initialize_arms(self, init_values):
        self.counts = [len(v) for v in init_values]
        return

    def select_arm(self):
        return random.choice(self.arms[self.valid == 1])

    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] = self.counts[chosen_arm] + 1
        if self.counts[chosen_arm] >= self.draw_limits[chosen_arm]:
            self.valid[chosen_arm] = 0
        return

    def update_batch(self, chosen_arm, reward):
        self.counts[chosen_arm] = self.counts[chosen_arm] + len(list(reward))
        if self.counts[chosen_arm] >= self.draw_limits[chosen_arm]:
            self.valid[chosen_arm] = 0
        return


class UCB_ThreshHard():
    def __init__(self, n_arms, draw_limits, p):
        self.past_values = [[] for col in range(n_arms)]
        self.sorted_values = []
        self.valid = np.ones(n_arms, dtype=np.int32)
        self.arms = np.arange(n_arms)
        self.draw_limits = draw_limits
        self.p = p
        self.thresh = -1.0
        return

    def initialize_arms(self, init_values):
        self.past_values = init_values
        self.sorted_values = list(np.sort(np.hstack(self.past_values)))
        self.thresh = self.sorted_values[-(int(len(self.sorted_values) * self.p) + 1)]
        return

    def select_arm(self):
        n_arms = len(self.arms)
        for arm in self.arms[self.valid == 1]:
            if len(self.past_values[arm]) == 0:
                return arm

        ucb_values = [0.0 for arm in self.arms]

        for arm in self.arms[self.valid == 1]:
            above_thresh = sum(np.array(self.past_values[arm]) >= self.thresh)
            ucb_values[arm] = above_thresh / len(self.past_values[arm])
        if sum(ucb_values) > 0.0:
            return ucb_values.index(max(ucb_values))
        else:
            max_values = [-1.0 for arm in self.arms]
            for arm in self.arms[self.valid == 1]:
                max_values[arm] = max(self.past_values[arm])
            return max_values.index(max(max_values))

    def update(self, chosen_arm, reward):
        self.past_values[chosen_arm].append(reward)
        if len(self.past_values[chosen_arm]) >= self.draw_limits[chosen_arm]:
            self.valid[chosen_arm] = 0

        bisect.insort(self.sorted_values, reward)
        self.thresh = self.sorted_values[-(int(len(self.sorted_values) * self.p) + 1)]
        return self.thresh

    def update_batch(self, chosen_arm, reward):
        for i in range(len(list(reward))):
            self.past_values[chosen_arm].append(reward[i])
            if len(self.past_values[chosen_arm]) >= self.draw_limits[chosen_arm]:
                self.valid[chosen_arm] = 0

            bisect.insort(self.sorted_values, reward[i])
            self.thresh = self.sorted_values[-(int(len(self.sorted_values) * self.p) + 1)]
        return self.thresh


class UCB_ThreshNormal():
    def __init__(self, n_arms, draw_limits, p):
        self.past_values = [[] for col in range(n_arms)]
        self.sorted_values = []
        self.valid = np.ones(n_arms, dtype=np.int32)
        self.arms = np.arange(n_arms)
        self.draw_limits = draw_limits
        self.p = p
        self.thresh = -1.0
        return

    def initialize_arms(self, init_values):
        self.past_values = init_values
        self.sorted_values = list(np.sort(np.hstack(self.past_values)))
        self.thresh = self.sorted_values[-(int(len(self.sorted_values) * self.p) + 1)]
        return

    def select_arm(self):
        n_arms = len(self.arms)
        for arm in self.arms[self.valid == 1]:
            if len(self.past_values[arm]) == 0:
                return arm

        ucb_values = [0.0 for arm in self.arms]

        for arm in self.arms[self.valid == 1]:
            arm_mu = np.mean(self.past_values[arm])
            arm_sigma = np.std(self.past_values[arm])
            ucb_values[arm] = 1 - Normal(mu=arm_mu, sigma=arm_sigma).cdf(self.thresh)
        if sum(ucb_values) > 0.0:
            return ucb_values.index(max(ucb_values))
        else:
            max_values = [-1.0 for arm in self.arms]
            for arm in self.arms[self.valid == 1]:
                max_values[arm] = max(self.past_values[arm])
            return max_values.index(max(max_values))

    def update(self, chosen_arm, reward):
        self.past_values[chosen_arm].append(reward)
        if len(self.past_values[chosen_arm]) >= self.draw_limits[chosen_arm]:
            self.valid[chosen_arm] = 0

        bisect.insort(self.sorted_values, reward)
        self.thresh = self.sorted_values[-(int(len(self.sorted_values) * self.p) + 1)]
        return self.thresh

    def update_batch(self, chosen_arm, reward):
        for i in range(len(list(reward))):
            self.past_values[chosen_arm].append(reward[i])
            if len(self.past_values[chosen_arm]) >= self.draw_limits[chosen_arm]:
                self.valid[chosen_arm] = 0

            bisect.insort(self.sorted_values, reward[i])
            self.thresh = self.sorted_values[-(int(len(self.sorted_values) * self.p) + 1)]
        return self.thresh


class UCB_Beta:
    def __init__(self, n_arms, draw_limits, beta):
        self.past_values = [[] for col in range(n_arms)]
        self.valid = np.ones(n_arms, dtype=np.int32)
        self.arms = np.arange(n_arms)
        self.draw_limits = draw_limits
        self.beta = beta
        return

    def initialize_arms(self, init_values):
        self.past_values = init_values
        return

    def select_arm(self):
        n_arms = len(self.arms)
        for arm in self.arms[self.valid == 1]:
            if len(self.past_values[arm]) == 0:
                return arm

        ucb_values = [0.0 for arm in self.arms]

        for arm in self.arms[self.valid == 1]:
            mean = np.mean(self.past_values[arm])
            bonus = np.std(self.past_values[arm])
            ucb_values[arm] = mean + self.beta * bonus
        return ucb_values.index(max(ucb_values))

    def update(self, chosen_arm, reward):
        self.past_values[chosen_arm].append(reward)
        if len(self.past_values[chosen_arm]) >= self.draw_limits[chosen_arm]:
            self.valid[chosen_arm] = 0
        return

    def update_batch(self, chosen_arm, rewards):
        self.past_values[chosen_arm].extend(list(rewards))
        if len(self.past_values[chosen_arm]) >= self.draw_limits[chosen_arm]:
            self.valid[chosen_arm] = 0
        return


class Cluster:
    def __init__(self, samples, dataset):
        self.samples = np.array(samples)
        self.dataset = dataset
        self.selected = np.zeros(len(self.samples))
        self.scores = np.ones(len(self.samples)) * -100.0

    def draw(self, model, optimizer_state, proj, val_grad, selected=True):
        remain_index = np.nonzero(self.selected == 0)[0]
        assert len(remain_index) > 0
        chosen = random.choice(remain_index)
        train_grad = get_training_gradients(self.dataset.select([chosen]), model, optimizer_state, proj)
        self.scores[chosen] = matching(train_grad, val_grad)[0]
        # print(self.scores[chosen])
        if selected:
            self.selected[chosen] = 1
        return self.scores[chosen]

    def draw_batch(self, num, model, optimizer_state, proj, val_grad, selected=True):
        remain_index = np.nonzero(self.selected == 0)[0]
        assert len(remain_index) > 0
        chosen = random.sample(list(remain_index), min(num, len(remain_index)))
        train_grad = get_training_gradients(self.dataset.select(chosen), model, optimizer_state, proj)
        self.scores[chosen] = matching(train_grad, val_grad)
        # print(self.scores[chosen])
        if selected:
            self.selected[chosen] = 1
        return self.scores[chosen]

    def size(self):
        return len(self.samples)

    def remain_size(self):
        return len(self.samples[self.selected == 0])

    def selected_subset(self):
        return self.samples[self.selected == 1], self.scores[self.selected == 1]

    def selected_mean(self):
        return np.mean(self.scores[self.selected == 1])

    def selected_std(self):
        return np.std(self.scores[self.selected == 1])


def ucb_selection(clusters, horizon, model, optimizer_state, proj, val_grad, method='ucb1', batch=1, beta=None, p=None, cold_start_p=0.0):
    # Initialise variables for duration of accumulated simulation (num_sims * horizon_per_simulation)
    num_batch = int(horizon/batch)
    chosen_clusters = [0.0 for i in range(num_batch)]
    chosen_scores = [0.0 for i in range(num_batch)]
    cumulative_scores = [0 for i in range(num_batch)]

    if method == 'ucb1':
        ucb = UCB1(len(clusters), [c.size() for c in clusters])
    elif method.startswith('ucb_beta'):
        assert beta is not None
        ucb = UCB_Beta(len(clusters), [c.size() for c in clusters], beta)
    elif method == 'ucb_threshhard':
        assert p is not None
        ucb = UCB_ThreshHard(len(clusters), [c.size() for c in clusters], p)
    elif method == 'ucb_threshnormal':
        assert p is not None
        ucb = UCB_ThreshNormal(len(clusters), [c.size() for c in clusters], p)

    if cold_start_p:
        cold_start_values = []
        for i in tqdm.tqdm(range(len(clusters)), desc="Cold start"):
            c = clusters[i]
            cold_start_values.append(list(c.draw_batch(max(int(c.size() * cold_start_p), 1), model, optimizer_state, proj, val_grad, selected=True)))
        ucb.initialize_arms(cold_start_values)

    for t in tqdm.tqdm(range(num_batch), desc="UCB selection"):
        # Selection of best cluster and engaging it
        chosen_cluster = ucb.select_arm()
        chosen_clusters[t] = chosen_cluster

        # Engage chosen Arm and obtain reward info
        rewards = clusters[chosen_cluster].draw_batch(batch, model, optimizer_state, proj, val_grad)
        chosen_scores[t] = sum(rewards)

        if t == 0:
            cumulative_scores[t] = sum(rewards)
        else:
            cumulative_scores[t] = cumulative_scores[t - 1] + sum(rewards)

        ucb.update_batch(chosen_cluster, rewards)

    if horizon % batch:
        chosen_cluster = ucb.select_arm()
        chosen_clusters.append(chosen_cluster)

        # Engage chosen Arm and obtain reward info
        rewards = clusters[chosen_cluster].draw_batch(horizon % batch, model, optimizer_state, proj, val_grad)
        chosen_scores.append(sum(rewards))

        cumulative_scores.append(cumulative_scores[-1] + sum(rewards))

        ucb.update_batch(chosen_cluster, rewards)

    return clusters, chosen_clusters, chosen_scores, cumulative_scores


def cluster_ucb_selection(args, verbose=True):
    random.seed(args.seed)
    # load model, tokenzier and optimizer state
    if verbose:
        print("Loading model, toknizer and optimizer......")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    with open(os.path.join(args.model_path, 'tokenizer_config.json'), 'r') as f:
        tokenizer_config = json.load(f)
        if tokenizer_config['tokenizer_class'] == "Qwen2Tokenizer":
            qwen = True
        else:
            qwen = False
    dtype = torch.float16 if args.torch_dtype == "float16" else torch.bfloat16
    model = load_model(args.model_path, args.base_model_path, dtype)

    # pad token is not added by default for pretrained models
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})

    # resize embeddings if needed (e.g. for LlamaTokenizer)
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    if args.initialize_lora:
        assert not isinstance(model, PeftModel)
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.lora_target_modules,
        )
        model = get_peft_model(model, lora_config)

    if isinstance(model, PeftModel):
        model.print_trainable_parameters()

    optimizer_path = os.path.join(args.model_path, "optimizer.bin")
    adam_optimizer_state = torch.load(optimizer_path, map_location="cpu")["state"]

    # build data dictionary
    if verbose:
        print("Loading training data samples......")
    total_candidates = build_data_list(args)
    train_files = [args.train_file_path.format(file) for file in args.train_file_names]
    train_dataset = get_training_dataset(train_files, tokenizer, args.max_length, sample_percentage=1.0, qwen=qwen)
    columns = deepcopy(train_dataset.column_names)
    columns.remove("input_ids")
    columns.remove("labels")
    columns.remove("attention_mask")
    train_dataset = train_dataset.remove_columns(columns)

    # k-means clustering
    if verbose:
        print(f"Loading initial K-mean Clustering from path {args.cluster_path}......")
    cl, c = torch.load(args.cluster_path)

    # build cluster objects
    if verbose:
        print("Building cluster objects......")
    clusters = []
    for i in range(args.k):
        cluster_samples = np.array(total_candidates)[cl == i]
        cluster_encoded_samples = train_dataset.select(np.arange(len(cl))[cl == i])
        clusters.append(Cluster(cluster_samples, cluster_encoded_samples))

    # prepare projector
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    projector = get_trak_projector(device)
    number_of_params = get_number_of_params(model)
    proj = projector(grad_dim=number_of_params,
                     proj_dim=args.gradient_projection_dimension,
                     seed=0,
                     proj_type=ProjectionType.rademacher,
                     device=device,
                     dtype=dtype,
                     block_size=128,
                     max_batch_size=8)

    # load validation gradients
    if verbose:
        print("Loading validation sample gradients......")
    val_grad = torch.load(args.grad_path.format(args.task, args.ckpt, args.task))

    # select cluster using ucb
    if verbose:
        print(f"Drawing clusters with {args.method}......")
    max_sampling = int(len(total_candidates) * (args.budget-args.csp))
    updated_clusters, chosen_clusters, chosen_scores, cumulative_scores = ucb_selection(clusters, max_sampling,
                                                                                        model, adam_optimizer_state,
                                                                                        proj,
                                                                                        val_grad,
                                                                                        method=args.method,
                                                                                        batch=args.draw_batch,
                                                                                        beta=args.beta, p=args.ucb_p,
                                                                                        cold_start_p=args.csp)

    # select top p samples from ucb selected samples
    if verbose:
        print(f"Selecting top {args.top_p * 100}% samples from sampled subset......")
    ucb_sample_scores = []
    ucb_sample_ids = []
    for c in updated_clusters:
        c_subset = c.selected_subset()
        ucb_sample_ids.extend(list(c_subset[0]))
        ucb_sample_scores.extend(list(c_subset[1]))
    assert len(set(ucb_sample_ids)) == len(ucb_sample_ids)
    top_scores, top_indice = torch.topk(torch.tensor(ucb_sample_scores), int(len(total_candidates) * args.top_p),
                                        largest=True)
    cluster_selected_samples = list(np.array(ucb_sample_ids)[top_indice])
    if verbose:
        print(f"Selection done: {len(cluster_selected_samples)} samples")

    return cluster_selected_samples, top_scores, updated_clusters


def write_candidates(candidates, train_file_path, output_path):
    candidates = sorted(candidates)
    candidate_files = [c.split('//')[0] for c in candidates]
    candidate_ids = [int(c.split('//')[1]) for c in candidates]
    unique_files, unique_indice = np.unique(candidate_files, return_index=True)
    print(unique_files, unique_indice)
    selected_lines = []
    for i, file in enumerate(unique_files):
        if file == 'gsm':
            train_file = train_file_path.format('gsm_train')
        else:
            train_file = train_file_path.format(file)
        with open(train_file, 'r') as fout:
            for j, line in enumerate(fout.readlines()):
                if i < len(unique_files)-1:
                    if j in candidate_ids[unique_indice[i]:unique_indice[i+1]]:
                        selected_lines.append(line)
                else:
                    if j in candidate_ids[unique_indice[i]:]:
                        selected_lines.append(line)

    with open(output_path, 'w') as fin:
        for line in selected_lines:
            fin.write(line)


def write_selected_data(output_path, cluster_selected_samples, top_scores):
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    sorted_score_file = os.path.join(output_path, 'sorted.csv')
    with open(sorted_score_file, 'w', encoding='utf-8') as file:
        file.write("file name, index, score\n")
        for sample_id, score in zip(cluster_selected_samples, top_scores):
            file.write(
                f"{sample_id.split('//')[0]}, {sample_id.split('//')[-1]}, {round(score.item(), 6)}\n")

    write_candidates(cluster_selected_samples, args.train_file_path, os.path.join(output_path, f"top_p{args.top_p}.jsonl"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, help="The name of the target task")
    parser.add_argument('--train_file_names', type=str, nargs='+', help='The name of the training file')
    parser.add_argument('--train_file_path', type=str, help='The path of the training data file')
    parser.add_argument('--model_path', type=str, help='the path of the model and tokenizer')
    parser.add_argument("--base_model_path", type=str, default=None, help="The path to the base model")
    parser.add_argument("--torch_dtype", type=str, default="float16", choices=["float16", "bfloat16"], help="The torch data type")
    parser.add_argument("--initialize_lora", default=False, action="store_true", help="Whether to initialize the base model with lora, only works when is_peft is False")
    parser.add_argument("--lora_r", type=int, default=8, help="The value of lora_r hyperparameter")
    parser.add_argument("--lora_alpha", type=float, default=32, help="The value of lora_alpha hyperparameter")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="The value of lora_dropout hyperparameter")
    parser.add_argument("--lora_target_modules", nargs='+', default=[ "q_proj", "k_proj", "v_proj", "o_proj"], help="The list of lora_target_modules")

    parser.add_argument("--grad_path", type=str, default=None, help="The path of grads used for selection")
    parser.add_argument("--gradient_projection_dimension", type=int, default=8192, help="The dimension of the projection")
    parser.add_argument("--max_length", type=int, default=2048, help="The maximum length in gradient computation")
    parser.add_argument("--chat_format", type=str, default="tulu", help="The chat format")
    parser.add_argument("--no_use_chat_format", action='store_true', help="Whether to use chat format")

    parser.add_argument("--cluster_path", type=str, default=None, help="The path of clustering results")
    parser.add_argument("--output_path", type=str, default=None, help="The path to the output")

    parser.add_argument("--ckpt", type=int, default=None, help="The model checkpoint used to select data")
    parser.add_argument("--method", type=str, default='ucb_beta', help="The UCB method used to select data")
    parser.add_argument("--k", type=int, default=None, help="The number of clusters")
    parser.add_argument("--beta", type=float, default=1, help="The beta used in UCB_Beta method")
    parser.add_argument("--ucb_p", type=float, default=0.25, help="The percentage used in UCB_Thresh method")
    parser.add_argument("--csp", type=float, default=0.0, help="The percentage of data used for cold start")
    parser.add_argument("--budget", type=float, default=0.2, help="The computing budget")
    parser.add_argument("--top_p", type=float, default=0.05, help="The percentage of final selected data")
    parser.add_argument("--draw_batch", type=int, default=1, help="The batch size of drawing in UCB")

    parser.add_argument("--seed", type=int, default=42, help="The random seed")

    args = parser.parse_args()

    cluster_selected_samples, top_scores, updated_clusters = cluster_ucb_selection(args, verbose=True)
    write_selected_data(args.output_path, cluster_selected_samples, top_scores)
