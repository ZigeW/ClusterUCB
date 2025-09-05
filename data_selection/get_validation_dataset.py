import json
import os
import random
from typing import List, Tuple, Iterable, Dict
import gzip
import copy

import pandas as pd
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq, PreTrainedTokenizerBase

# llama-chat model's instruction format
B_INST, E_INST = "[INST]", "[/INST]"


def tokenize(tokenizer: PreTrainedTokenizerBase,
             query: str,
             completion: str,
             max_length: int,
             print_ex: bool = False) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    """
    Formats a chat conversation into input tensors for a transformer model.

    Args:
        tokenizer (PreTrainedTokenizerBase): The tokenizer used to encode the input.
        query (str): The question part of the chat conversation.
        completion (str): The answer part of the chat conversation.
        max_length (int): The maximum length of the input tensors.
        print_ex (bool, optional): Whether to print the example. Defaults to False.

    Returns:
        tuple: A tuple containing the full input IDs, labels, and attention mask tensors.
    """
    full_prompt = query + completion

    if print_ex:
        print("******** Example starts ********")
        print(full_prompt)
        print("******** Example ends ********")

    prompt_input_ids = torch.tensor(
        tokenizer.encode(query, max_length=max_length))
    full_input_ids = torch.tensor(
        tokenizer.encode(full_prompt, max_length=max_length))
    labels = torch.tensor(tokenizer.encode(full_prompt, max_length=max_length))
    labels[:len(prompt_input_ids)] = -100
    attention_mask = [1] * len(full_input_ids)

    return full_input_ids, labels, attention_mask


def stream_jsonl(filename: str) -> Iterable[Dict]:
    """
    Parses each jsonl line and yields it as a dictionary
    """
    if filename.endswith(".gz"):
        with open(filename, "rb") as gzfp:
            with gzip.open(gzfp, 'rt') as fp:
                for line in fp:
                    if any(not x.isspace() for x in line):
                        yield json.loads(line)
    else:
        with open(filename, "r") as fp:
            for line in fp:
                if any(not x.isspace() for x in line):
                    yield json.loads(line)


def create_prompt_with_huggingface_tokenizer_template(messages, tokenizer, add_bos=False):
    formatted_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if add_bos:
        formatted_text = tokenizer.bos_token + formatted_text
    return formatted_text


def get_bbh_dataset(data_dir: str,
                    tokenizer: PreTrainedTokenizerBase,
                    max_length: int,
                    use_chat_format: bool = True,
                    chat_format: str = "tulu",
                    **kwargs):
    """
    Get the bbh dataset in the instruction tuning format. Each example is formatted as follows: 

    Query: 
    <|user|>
    <Task Prompt>
    <Ex1>
    <Ex2>
    <Question of Ex3>
    <|assistant|>
    A:

    Completion:
    <Answer of Ex3>

    Args:
        data_dir (str): The main data directory.
        tokenizer (Tokenizer): The tokenizer used to tokenize the input text.
        max_length (int): The maximum length of the input sequence.
        use_chat_format (bool, optional): Whether to use chat format for the input. Defaults to True.
        chat_format (str, optional): The chat format to use. Defaults to "tulu".
        n_shot (int, optional): The number of shots for few-shot learning. Defaults to 3 for bbh.

    Returns:
        Dataset: The BBH dataset containing input_ids, attention_mask, and labels.
    """
    file = f"{data_dir}/eval/bbh/bbh-three-shot.json"

    bbh_few_shot_examples = json.load(open(file, "r"))
    dataset = {"input_ids": [], "attention_mask": [], "labels": []}

    # there are multiple tasks in the bbh dataset
    # each task has 3 examples
    for task in bbh_few_shot_examples:
        few_shot_exs = bbh_few_shot_examples[task]

        stuff = few_shot_exs.split("\n\n")
        exes = stuff[-3:]
        task_prompt = "\n\n".join(stuff[:-3])

        def form_icl(exs):
            string = ""
            for ex in exs:
                question, answer = ex.split("\nA:")
                string += question + "\nA:" + answer
                string += "\n\n"
            return string

        for i in range(len(exes)):
            target_ex = exes[i]
            other_exes = exes[:i] + exes[i+1:]
            icl = form_icl(other_exes)
            # icl = ""
            question, answer = target_ex.split("\nA:")

            if use_chat_format:
                if chat_format == "tulu":  # we follow the tulu instruction tuning format
                    question = "<|user|>\n" + task_prompt.strip() + "\n\n" + icl + \
                        f"{question}" + "\n<|assistant|>\nA:"
                elif chat_format == 'huggingface':
                    messages = [{"role": "user", "content": task_prompt.strip() + "\n\n" + icl + f"{question}"}]
                    question = create_prompt_with_huggingface_tokenizer_template(messages, tokenizer)
                    question += "A:" if question[-1] in ["\n", " "] else " A:"
                else:
                    question = f"<s> {B_INST} {task_prompt.strip()} {question} {E_INST} A:"
            else:
                question = task_prompt.strip() + "\n\n" + \
                    f"{question}" + "\nA:"
            full_input_ids, labels, attention_mask = tokenize(
                tokenizer, question, answer, max_length, print_ex=True)
            dataset["input_ids"].append(full_input_ids)
            dataset["labels"].append(labels)
            dataset["attention_mask"].append(attention_mask)

    dataset = Dataset.from_dict(dataset)
    return dataset


def get_tydiqa_dataset(data_dir: str,
                       tokenizer: PreTrainedTokenizerBase,
                       max_length: int,
                       use_chat_format: bool = True,
                       chat_format: str = "tulu",
                       zh: bool = False,
                       **kwargs) -> Dataset:
    """
    Get the tydiqa dataset in the instruction tuning format. Each example is formatted as follows:  

    Query: 
    <|user|>
    <Task Prompt>
    <Passage>
    <Question>
    <|assistant|>
    Answer: 

    Completion:
    <Answer>

    Args:
        data_dir (str): The main data directory.
        tokenizer (PreTrainedTokenizerBase): The tokenizer to use for tokenization.
        max_length (int): The maximum length of the input sequence.
        use_chat_format (bool, optional): Whether to use chat format. Defaults to True.
        chat_format (str, optional): The chat format to use. Defaults to "tulu".
        zh (bool, optional): Whether to use the Chinese validation examples. Defaults to False.

    Returns:
        Dataset: The tokenized TydiQA dataset.
    """

    # Same template as https://github.com/allenai/open-instruct/blob/main/eval/tydiqa/run_eval.py#L17
    encoding_templates_with_context = {
        "english": ("Answer the following question based on the information in the given passage.", "Passage:", "Question:", "Answer:"),
        "arabic": ("أجب على السؤال التالي بناءً على المعلومات في المقطع المعطى.", "المقطع:", "السؤال:", "الإجابة:"),
        "bengali": ("প্রদত্ত অধ্যায়ের তথ্যের উপর ভিত্তি করে নিম্নলিখিত প্রশ্নের উত্তর দিন।", "অধ্যায়:", "প্রশ্ন:", "উত্তর:"),
        "finnish": ("Vastaa seuraavaan kysymykseen annetun kappaleen tiedon perusteella.", "Kappale:", "Kysymys:", "Vastaus:"),
        "indonesian": ("Jawab pertanyaan berikut berdasarkan informasi di bagian yang diberikan.", "Bagian:", "Pertanyaan:", "Jawaban:"),
        "korean": ("주어진 문단의 정보에 기반하여 다음 질문에 답하십시오.", "문단:", "질문:", "답변:"),
        "russian": ("Ответьте на следующий вопрос на основе информации в данном отрывке.", "Отрывок:", "Вопрос:", "Ответ:"),
        "swahili": ("Jibu swali lifuatalo kulingana na habari kwenye kifungu kilichotolewa.", "Kifungu:", "Swali:", "Jibu:"),
        "telugu": ("ఇచ్చిన పేరాలోని సమాచారం ఆధారంగా కింది ప్రశ్నకు సమాధానం ఇవ్వండి.", "పేరా:", "ప్రశ్న:", "సమాధానం:")
    }

    # Chinese validation examples
    if zh:
        for lang in encoding_templates_with_context:
            encoding_templates_with_context[lang] = (
                "根据所给文章中的信息回答以下问题。", "文章:", "问题:", "答案:")

    file_name = "tydiqa-one-shot-zh.json" if zh else "tydiqa-one-shot.json"
    file = os.path.join(f"{data_dir}/eval/tydiqa", file_name)

    examples = json.load(open(file, "r"))
    dataset = {"input_ids": [], "attention_mask": [], "labels": []}

    for i, lang in enumerate(examples):
        example = examples[lang][0]
        prompt, p_template, q_template, a_template = encoding_templates_with_context[lang]
        prompt += p_template + " " + \
            format(example["context"]) + "\n" + q_template + \
            " " + format(example["question"]) + "\n"
        answer = " " + format(example["answers"][0]["text"])
        if use_chat_format:
            if chat_format == "tulu":
                prompt = "<|user|>\n" + prompt + "<|assistant|>\n" + a_template
            elif chat_format == 'huggingface':
                messages = [{"role": "user", "content": prompt}]
                prompt = create_prompt_with_huggingface_tokenizer_template(messages, tokenizer)
                prompt += a_template if prompt[-1] in ["\n", " "] else " " + a_template
            else:
                prompt = f"<s> {B_INST} {prompt} {E_INST} {a_template}"
        else:
            prompt = prompt + a_template
        full_input_ids, labels, attention_mask = tokenize(
            tokenizer, prompt, answer, max_length, print_ex=True)
        dataset["input_ids"].append(full_input_ids)
        dataset["labels"].append(labels)
        dataset["attention_mask"].append(attention_mask)
    dataset = Dataset.from_dict(dataset)
    return dataset


def get_mmlu_dataset(data_dir: str,
                     tokenizer: PreTrainedTokenizerBase,
                     max_length: int,
                     use_chat_format=True,
                     chat_format="tulu",
                     few_shot=False,
                     **kwargs):
    """
    Get the MMLU dataset in the instruction tuning format. Each example is formatted as follows:

    Query:
    <|user|>
    <Task Prompt>
    <Question>
    <|assistant|>
    The answer is:

    Completion:
    <Answer>

    Args:
        data_dir (str): The main data directory.
        tokenizer (Tokenizer): The tokenizer used to tokenize the input text.
        max_length (int): The maximum length of the input sequence.
        use_chat_format (bool, optional): Whether to use chat format for the prompts. Defaults to True.
        chat_format (str, optional): The chat format to use for the prompts. Defaults to "tulu".

    Returns:
        Dataset: The tokenized dataset containing input_ids, attention_mask, and labels.
    """
    choices = ["A", "B", "C", "D"]
    mmlu_data_dir = os.path.join(data_dir, "eval", "mmlu")
    subjects = sorted(
        [
            f.split("_test.csv")[0]
            for f in os.listdir(os.path.join(mmlu_data_dir, "test"))
            if "_test.csv" in f
        ]
    )

    def format_subject(subject):
        l = subject.split("_")
        s = ""
        for entry in l:
            s += " " + entry
        return s

    def gen_prompt(train_df, subject, k=0):
        prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
            format_subject(subject)
        )
        if k == -1:
            k = train_df.shape[0]
        for i in range(k):
            prompt += format_example(train_df, i)
        return prompt

    def format_example(df, idx, include_answer=True):
        prompt = df.iloc[idx, 0]
        k = df.shape[1] - 2
        for j in range(k):
            prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
        prompt += "\nAnswer:"
        if include_answer:
            prompt += " {}\n\n".format(df.iloc[idx, k + 1])
        return prompt

    k = 5
    dataset = {"input_ids": [], "attention_mask": [], "labels": []}
    for subject in subjects:
        dev_df = pd.read_csv(
            os.path.join(mmlu_data_dir, "dev", subject + "_dev.csv"), header=None
        )[: k]
        for i in range(k):
            if few_shot:
                prompt_end = format_example(dev_df, i, include_answer=False)
                dev_exp_df = copy.deepcopy(dev_df)
                train_prompt = gen_prompt(dev_exp_df.drop([i]), subject, k - 1)
                prompt = train_prompt + prompt_end
            else:
                prompt = gen_prompt(dev_df, subject, 0) + format_example(dev_df, i, include_answer=False)
            answer = " " + dev_df.iloc[i, dev_df.shape[1] - 2 + 1]

            if use_chat_format:
                if chat_format == "tulu":
                    prompt = "<|user|>\n" + prompt + "\n<|assistant|>\nThe answer is:"
                elif chat_format == 'huggingface':
                    messages = [{"role": "user", "content": prompt}]
                    prompt = create_prompt_with_huggingface_tokenizer_template(messages, tokenizer)
                    if prompt[-1] in ["\n", " "]:
                        prompt += "The answer is:"
                    else:
                        prompt += " The answer is:"
                else:
                    # f"<s> {B_INST} {task_prompt.strip()} {question} {E_INST} A:"
                    prompt = f"<s> {B_INST} {prompt} {E_INST} The answer is:"
            else:
                prompt = prompt
            full_input_ids, labels, attention_mask = tokenize(
                tokenizer, prompt, answer, max_length, print_ex=True if i == 0 else False)
            dataset["input_ids"].append(full_input_ids)
            dataset["labels"].append(labels)
            dataset["attention_mask"].append(attention_mask)
    dataset = Dataset.from_dict(dataset)
    return dataset


def get_codex_dataset(data_dir: str,
                       tokenizer: PreTrainedTokenizerBase,
                       max_length: int,
                       use_chat_format: bool = True,
                       chat_format: str = "tulu",
                       k: int = 10,
                       help: bool = True,
                       **kwargs) -> Dataset:
    """
    Get the HumanEval dataset in the instruction tuning format. Each example is formatted as follows:

    Query:
    <|user|>
    Complete the following python function.
    <Task Prompt>
    <Question>
    <|assistant|>
    Here is the completed function:

    Completion:
    <Answer>

    Args:
        data_dir (str): The main data directory.
        tokenizer (PreTrainedTokenizerBase): The tokenizer to use for tokenization.
        max_length (int): The maximum length of the input sequence.
        use_chat_format (bool, optional): Whether to use chat format. Defaults to True.
        chat_format (str, optional): The chat format to use. Defaults to "tulu".

    Returns:
        Dataset: The tokenized HumanEval dataset.
    """

    file_name = "HumanEval.jsonl.gz"
    file = os.path.join(f"{data_dir}/eval/codex-humaneval", file_name)
    examples = {task["task_id"]: task for task in stream_jsonl(file)}

    # k = 10
    random.seed(42)
    selected_example = random.sample(examples.items(), k)

    dataset = {"input_ids": [], "attention_mask": [], "labels": []}

    if help:
        help_file = os.path.join(f"{data_dir}/eval/codex-humaneval", 'humanevalpack.jsonl')
        with open(help_file, "r") as f:
            instructions = [json.loads(l) for l in f]
            instructions_dict = {
                x["task_id"].replace("Python", "HumanEval"): x["instruction"] for x in instructions
            }
        a_template = "Here is the function:\n\n```python\n"
    else:
        instructions_dict = None
        instruction = "Complete the following python function.\n\n\n"
        a_template = "Here is the completed function:\n\n\n```python\n"

    for idx, example in selected_example:
        if help:
            prompt = instructions_dict[example['task_id']]
            answer = example["canonical_solution"] + "\n```"
        else:
            prompt = instruction + example["prompt"]
            answer = example['canonical_solution'] + "\n```"
        if use_chat_format:
            if chat_format == "tulu":
                if help:
                    prompt = "<|user|>\n" + prompt + "\n<|assistant|>\n" + a_template + example["prompt"]
                else:
                    prompt = "<|user|>\n" + prompt + "<|assistant|>\n" + a_template
            elif chat_format == 'huggingface':
                messages = [{"role": "user", "content": prompt}]
                prompt = create_prompt_with_huggingface_tokenizer_template(messages, tokenizer)
                prefix = "" if prompt[-1] in ["\n", " "] else " "
                prompt = prompt + prefix + a_template + example["prompt"]
            else:
                prompt = f"<s> {B_INST} {prompt} {E_INST} {a_template}"
        else:
            prompt = example['prompt']
        full_input_ids, labels, attention_mask = tokenize(
            tokenizer, prompt, answer, max_length, print_ex=True)
        dataset["input_ids"].append(full_input_ids)
        dataset["labels"].append(labels)
        dataset["attention_mask"].append(attention_mask)
    dataset = Dataset.from_dict(dataset)
    return dataset


def get_gsm_dataset(data_dir: str,
                    tokenizer: PreTrainedTokenizerBase,
                    max_length: int,
                    use_chat_format: bool = True,
                    chat_format: str = "tulu",
                    k: int = 20,
                    replace_format: bool = True,
                    **kwargs) -> Dataset:
    """
    Get the GSM8k dataset in the instruction tuning format. Each example is formatted as follows:

    Query:
    <|user|>
    Answer the following question.
    Question: <Question>
    <|assistant|>
    Answer:

    Completion:
    <Answer>

    Args:
        data_dir (str): The main data directory.
        tokenizer (PreTrainedTokenizerBase): The tokenizer to use for tokenization.
        max_length (int): The maximum length of the input sequence.
        use_chat_format (bool, optional): Whether to use chat format. Defaults to True.
        chat_format (str, optional): The chat format to use. Defaults to "tulu".

    Returns:
        Dataset: The tokenized GSM8k dataset.
    """

    # file_name = "examplars.json"
    # file = os.path.join(f"{data_dir}/eval/gsm", file_name)
    # with open(file, 'r') as f:
    #     examples = json.load(f)
    file = "../data/gsm/test.jsonl"
    examples = []
    with open(file, 'r') as f:
        for line in f.readlines():
            example = json.loads(line)
            cot, answer = example["answer"].split("\n####")
            if replace_format:
                if not cot.endswith('.'):
                    cot += '.'
                if not answer.endswith('.'):
                    answer += '.'
                examples.append({
                    "question": example["question"],
                    "answer": cot + ' So the answer is' + answer
                })
            else:
                examples.append(example)

    # k = 20
    random.seed(42)
    examples = random.sample(examples, k)

    dataset = {"input_ids": [], "attention_mask": [], "labels": []}

    instruction = "Answer the following question.\n\nQuestion: "
    a_template = "Answer: \n\n"
    for example in examples:
        prompt = instruction + example["question"] + '\n'
        # cot, answer = example['cot_answer'].split("So the answer is ")
        # cot = cot + "So the answer is "
        # answer = example['cot_answer'].replace(" So the answer is", "\n####")
        answer = example['answer']
        if use_chat_format:
            if chat_format == "tulu":
                prompt = "<|user|>\n" + prompt + "<|assistant|>\n" + a_template
            elif chat_format == 'huggingface':
                messages = [{"role": "user", "content": prompt}]
                prompt = create_prompt_with_huggingface_tokenizer_template(messages, tokenizer)
                prompt += "Answer:" if prompt[-1] in ["\n", " "] else " Answer:"
            else:
                prompt = f"<s> {B_INST} {prompt} {E_INST} {a_template}"
        else:
            prompt = prompt + a_template
        full_input_ids, labels, attention_mask = tokenize(
            tokenizer, prompt, answer, max_length, print_ex=True)
        dataset["input_ids"].append(full_input_ids)
        dataset["labels"].append(labels)
        dataset["attention_mask"].append(attention_mask)
    dataset = Dataset.from_dict(dataset)
    return dataset


def get_dataset(task, **kwargs):
    """
    Get the dataset for the given task.

    Args:
        task_name (str): The name of the task.

    Raises:
        ValueError: If the task name is not valid.

    Returns:
        Dataset: The dataset.
    """
    if task == "bbh":
        return get_bbh_dataset(**kwargs)
    elif task == "tydiqa":
        return get_tydiqa_dataset(**kwargs)
    elif task == "mmlu":
        return get_mmlu_dataset(**kwargs, few_shot=False)
    elif task == 'humaneval':
        return get_codex_dataset(**kwargs, k=10)
    elif task == 'gsm':
        return get_gsm_dataset(**kwargs, k=50, replace_format=False)
    else:
        raise ValueError("Invalid task name")


def get_dataloader(dataset, tokenizer, batch_size=1):
    data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer, padding="longest") 
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,  # When getting gradients, we only do this single batch process
                            collate_fn=data_collator)
    print("There are {} examples in the dataset".format(len(dataset)))
    return dataloader
