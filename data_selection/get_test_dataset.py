import json
import os
import random
from typing import List, Tuple, Iterable, Dict
import gzip
import glob
import re

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


def create_prompt_with_tulu_chat_format(messages, tokenizer, bos="<s>", eos="</s>", add_bos=True):
    formatted_text = ""
    for message in messages:
        if message["role"] == "system":
            formatted_text += "<|system|>\n" + message["content"] + "\n"
        elif message["role"] == "user":
            formatted_text += "<|user|>\n" + message["content"] + "\n"
        elif message["role"] == "assistant":
            formatted_text += "<|assistant|>\n" + message["content"].strip() + eos + "\n"
        else:
            raise ValueError(
                "Tulu chat template only supports 'system', 'user' and 'assistant' roles. Invalid role: {}.".format(message["role"])
                )
    formatted_text += "<|assistant|>\n"
    formatted_text = bos + formatted_text if add_bos else formatted_text
    return formatted_text


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
    data_dir = os.path.join(data_dir, 'bbh')
    dataset = {"input_ids": [], "attention_mask": [], "labels": []}

    all_tasks = {}
    task_files = glob.glob(os.path.join(data_dir, "bbh", "*.json"))
    for task_file in task_files:
        with open(task_file, "r") as f:
            task_name = os.path.basename(task_file).split(".")[0]
            all_tasks[task_name] = json.load(f)["examples"]
            # if max_num_examples_per_task:
            #     all_tasks[task_name] = random.sample(all_tasks[task_name], max_num_examples_per_task)

    all_prompts = {}
    cot_prompt_files = glob.glob(os.path.join(data_dir, "cot-prompts", "*.txt"))
    for cot_prompt_file in cot_prompt_files:
        with open(cot_prompt_file, "r") as f:
            task_name = os.path.basename(cot_prompt_file).split(".")[0]
            task_prompt = "".join(f.readlines()[2:])
            all_prompts[task_name] = task_prompt

    assert set(all_tasks.keys()) == set(
        all_prompts.keys()), "task names in task data and task prompts are not the same."

    prompts = []
    for task_name in all_tasks.keys():
        task_examples = all_tasks[task_name]
        task_prompt = all_prompts[task_name]
        # prepare prompts
        if use_chat_format:
            for example in task_examples:
                prompt = task_prompt.strip() + "\n\nQ: " + example["input"]
                messages = [{"role": "user", "content": prompt}]
                if chat_format == "tulu":
                    prompt = create_prompt_with_tulu_chat_format(messages, tokenizer, add_bos=False)
                else:
                    prompt = create_prompt_with_huggingface_tokenizer_template(messages, tokenizer)
                prompt += "A:" if prompt[-1] in ["\n", " "] else " A:"
                prompts.append((prompt, " " + example['target']))
        else:
            prompts.extend([(task_prompt.strip() + "\n\nQ: " + example["input"] + "\nA:", " " + example['target']) for example in task_examples])

    for question, answer in prompts:
        full_input_ids, labels, attention_mask = tokenize(
            tokenizer, question, answer, max_length, print_ex=False)
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

    data_dir = os.path.join(data_dir, 'tydiqa')
    dataset = {"input_ids": [], "attention_mask": [], "labels": []}
    random.seed(42)

    test_data = []
    with open(os.path.join(data_dir, "tydiqa-goldp-v1.1-dev.json")) as fin:
        dev_data = json.load(fin)
        for article in dev_data["data"]:
            for paragraph in article["paragraphs"]:
                for qa in paragraph["qas"]:
                    example = {
                        "id": qa["id"],
                        "lang": qa["id"].split("-")[0],
                        "context": paragraph["context"],
                        "question": qa["question"],
                        "answers": qa["answers"]
                    }
                    test_data.append(example)
    data_languages = sorted(list(set([example["lang"] for example in test_data])))

    print(f"Loaded {len(test_data)} examples from {len(data_languages)} languages: {data_languages}")

    train_data_for_langs = {lang: [] for lang in data_languages}
    with open(os.path.join(data_dir, "tydiqa-goldp-v1.1-train.json")) as fin:
        train_data = json.load(fin)
        for article in train_data["data"]:
            for paragraph in article["paragraphs"]:
                for qa in paragraph["qas"]:
                    lang = qa["id"].split("-")[0]
                    if lang in data_languages:
                        example = {
                            "id": qa["id"],
                            "lang": lang,
                            "context": paragraph["context"],
                            "question": qa["question"],
                            "answers": qa["answers"]
                        }
                        train_data_for_langs[lang].append(example)
        for lang in data_languages:
            # sample n_shot examples from each language
            train_data_for_langs[lang] = random.sample(train_data_for_langs[lang], 1)
        # assert that we have exactly n_shot examples for each language
        assert all([len(train_data_for_langs[lang]) == 1 for lang in data_languages])

    # assert we have templates for all data languages
    assert all([lang in encoding_templates_with_context.keys() for lang in data_languages])

    if max_length:
        for example in test_data:
            tokenized_context = tokenizer.encode(example["context"])
            if len(tokenized_context) > max_length:
                example["context"] = tokenizer.decode(tokenized_context[:max_length])
        for lang in data_languages:
            for example in train_data_for_langs[lang]:
                tokenized_context = tokenizer.encode(example["context"])
                if len(tokenized_context) > max_length:
                    example["context"] = tokenizer.decode(tokenized_context[:max_length])

    prompts = []
    for example in test_data:
        lang = example["lang"]
        prompt, p_template, q_template, a_template = encoding_templates_with_context[lang]
        prompt += "\n\n"

        formatted_demo_examples = []
        for train_example in train_data_for_langs[lang]:
            formatted_demo_examples.append(
                p_template + " " + train_example["context"] + "\n" + q_template + " " + train_example[
                    "question"] + "\n" + a_template + " " + train_example["answers"][0]["text"]
            )
        prompt += "\n\n".join(formatted_demo_examples) + "\n\n"

        prompt += p_template + " " + format(example["context"]) + "\n" + q_template + " " + format(
                example["question"]) + "\n"

        if use_chat_format:
            messages = [{"role": "user", "content": prompt}]
            if chat_format == "tulu":
                prompt = create_prompt_with_tulu_chat_format(messages, tokenizer, add_bos=False)
            else:
                prompt = create_prompt_with_huggingface_tokenizer_template(messages, tokenizer)
            prompt += a_template if prompt[-1] in ["\n", " "] else " " + a_template
        else:
            prompt += a_template
        prompts.append((prompt, " " + example['answers'][0]['text']))

    for prompt, answer in prompts:
        full_input_ids, labels, attention_mask = tokenize(
            tokenizer, prompt, answer, max_length, print_ex=False)
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
                     **kwargs):
    """
    Args:
        data_dir (str): The main data directory.
        tokenizer (Tokenizer): The tokenizer used to tokenize the input text.
        max_length (int): The maximum length of the input sequence.
        use_chat_format (bool, optional): Whether to use chat format for the prompts. Defaults to True.
        chat_format (str, optional): The chat format to use for the prompts. Defaults to "tulu".

    Returns:
        Dataset: The tokenized dataset containing input_ids, attention_mask, and labels.
    """
    data_dir = os.path.join(data_dir, "mmlu")
    subjects = sorted(
        [
            f.split("_test.csv")[0]
            for f in os.listdir(os.path.join(data_dir, "test"))
            if "_test.csv" in f
        ]
    )
    choices = ["A", "B", "C", "D"]

    def format_subject(subject):
        l = subject.split("_")
        s = ""
        for entry in l:
            s += " " + entry
        return s

    def format_example(df, idx, include_answer=True):
        prompt = df.iloc[idx, 0]
        k = df.shape[1] - 2
        for j in range(k):
            prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
        prompt += "\nAnswer:"
        if include_answer:
            prompt += " {}\n\n".format(df.iloc[idx, k + 1])
        return prompt

    def gen_prompt(train_df, subject, k=-1):
        prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
            format_subject(subject)
        )
        if k == -1:
            k = train_df.shape[0]
        for i in range(k):
            prompt += format_example(train_df, i)
        return prompt

    dataset = {"input_ids": [], "attention_mask": [], "labels": []}
    prompts = []
    for subject in subjects:
        k = 5
        dev_df = pd.read_csv(
            os.path.join(data_dir, "dev", subject + "_dev.csv"), header=None
        )[: k]
        test_df = pd.read_csv(
            os.path.join(data_dir, "test", subject + "_test.csv"), header=None
        )
        for i in range(0, test_df.shape[0]):
            prompt_end = format_example(test_df, i, include_answer=False)
            train_prompt = gen_prompt(dev_df, subject, k)
            prompt = train_prompt + prompt_end

            if use_chat_format:
                messages = [{"role": "user", "content": prompt}]
                if chat_format == "tulu":
                    prompt = create_prompt_with_tulu_chat_format(messages, tokenizer, add_bos=False)
                else:
                    prompt = create_prompt_with_huggingface_tokenizer_template(messages, tokenizer)
                if prompt[-1] in ["\n", " "]:
                    prompt += "The answer is:"
                else:
                    prompt += " The answer is:"

            tokenized_prompt = tokenizer(prompt, truncation=False, add_special_tokens=False).input_ids
            # make sure every prompt is less than 2048 tokens
            while len(tokenized_prompt) > 2048:
                k -= 1
                train_prompt = gen_prompt(dev_df, subject, k)
                prompt = train_prompt + prompt_end

                if use_chat_format:
                    messages = [{"role": "user", "content": prompt}]
                    if chat_format == "tulu":
                        prompt = create_prompt_with_tulu_chat_format(messages, tokenizer, add_bos=False)
                    else:
                        prompt = create_prompt_with_huggingface_tokenizer_template(messages, tokenizer)
                    if prompt[-1] in ["\n", " "]:
                        prompt += "The answer is:"
                    else:
                        prompt += " The answer is:"

                tokenized_prompt = tokenizer(prompt, truncation=False, add_special_tokens=False).input_ids
            prompts.append((prompt, " " + test_df.iloc[i, -1]))

    for prompt, answer in prompts:
        full_input_ids, labels, attention_mask = tokenize(
            tokenizer, prompt, answer, max_length, print_ex=False)
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
                       **kwargs) -> Dataset:
    """
    Get the HumanEval dataset in the instruction tuning format. Each example is formatted as follows:
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
    file = os.path.join(f"{data_dir}/codex-humaneval", file_name)
    test_data = {task["task_id"]: task for task in stream_jsonl(file)}
    test_data = list(test_data.values())

    dataset = {"input_ids": [], "attention_mask": [], "labels": []}

    help_file = os.path.join(f"{data_dir}/codex-humaneval", 'humanevalpack.jsonl')
    with open(help_file, "r") as f:
        instructions = [json.loads(l) for l in f]
        instructions_dict = {
            x["task_id"].replace("Python", "HumanEval"): x["instruction"] for x in instructions
        }
    answer = "Here is the function:\n\n```python\n"

    if use_chat_format:
        prompts = []
        for example in test_data:
            instruction = instructions_dict[example["task_id"]]
            messages = [{"role": "user", "content": instruction}]
            if chat_format == "tulu":
                prompt = create_prompt_with_tulu_chat_format(messages, tokenizer, add_bos=False)
            else:
                prompt = create_prompt_with_huggingface_tokenizer_template(messages, tokenizer)
            prefix = "" if prompt[-1] in ["\n", " "] else " "
            prompt = prompt + prefix + answer + example["prompt"]
            prompts.append((prompt, example["canonical_solution"] + "\n```"))
    else:
        prompts = [(example["prompt"], example["canonical_solution"] + "\n```") for example in test_data]

    for prompt, answer in prompts:
        full_input_ids, labels, attention_mask = tokenize(
            tokenizer, prompt, answer, max_length, print_ex=False)
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
                       **kwargs) -> Dataset:
    """
    Get the GSM8k dataset in the instruction tuning format. Each example is formatted as follows:
    Args:
        data_dir (str): The main data directory.
        tokenizer (PreTrainedTokenizerBase): The tokenizer to use for tokenization.
        max_length (int): The maximum length of the input sequence.
        use_chat_format (bool, optional): Whether to use chat format. Defaults to True.
        chat_format (str, optional): The chat format to use. Defaults to "tulu".

    Returns:
        Dataset: The tokenized GSM8k dataset.
    """
    file = os.path.join(data_dir, "gsm/test.jsonl")
    test_data = []
    with open(file) as fin:
        for line in fin:
            example = json.loads(line)
            cot, answer = example["answer"].split("\n####")
            if not cot.endswith('.'):
                cot += '.'
            if not answer.endswith('.'):
                answer += '.'
            test_data.append({
                "question": example["question"],
                "answer": cot + ' So the answer is' + answer
            })
    # for example in test_data:
    #     example["answer"] = re.sub(r"(\d),(\d)", r"\1\2", example["answer"])
    #     assert float(example["answer"]), f"answer is not a valid number: {example['answer']}"

    dataset = {"input_ids": [], "attention_mask": [], "labels": []}

    GSM_EXAMPLARS = [
    {
        "question": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
        "cot_answer": "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. So the answer is 6.",
        "short_answer": "6"
    },
    {
        "question": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
        "cot_answer": "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. So the answer is 5.",
        "short_answer": "5"
    },
    {
        "question": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
        "cot_answer": "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. So the answer is 39.",
        "short_answer": "39"
    },
    {
        "question": "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
        "cot_answer": "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. So the answer is 8.",
        "short_answer": "8"
    },
    {
        "question": "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
        "cot_answer": "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. So the answer is 9.",
        "short_answer": "9"
    },
    {
        "question": "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
        "cot_answer": "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. So the answer is 29.",
        "short_answer": "29"
    },
    {
        "question": "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
        "cot_answer": "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. So the answer is 33.",
        "short_answer": "33"
    },
    {
        "question": "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
        "cot_answer": "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. So the answer is 8.",
        "short_answer": "8"
    }
]
    demonstrations = []
    for example in GSM_EXAMPLARS:
        demonstrations.append(
            "Question: " + example["question"] + "\n" + "Answer: " + example["cot_answer"]
        )
    prompt_prefix = "Answer the following questions.\n\n" + "\n\n".join(demonstrations) + "\n\n"

    if use_chat_format:
        prompts = []
        for example in test_data:
            messages = [{"role": "user", "content": prompt_prefix + "Question: " + example["question"].strip()}]
            if chat_format == "tulu":
                prompt = create_prompt_with_tulu_chat_format(messages, tokenizer, add_bos=False)
            else:
                prompt = create_prompt_with_huggingface_tokenizer_template(messages, tokenizer)
            prompt += "Answer:" if prompt[-1] in ["\n", " "] else " Answer:"
            prompts.append((prompt, " " + example["answer"]))
    else:
        prompts = [(prompt_prefix + "Question: " + example["question"].strip() + "\nAnswer:", " " + example["answer"]) for example in test_data]

    for prompt, answer in prompts:
        full_input_ids, labels, attention_mask = tokenize(
            tokenizer, prompt, answer, max_length, print_ex=False)
        dataset["input_ids"].append(full_input_ids)
        dataset["labels"].append(labels)
        dataset["attention_mask"].append(attention_mask)
    dataset = Dataset.from_dict(dataset)
    return dataset


def get_test_dataset(task, **kwargs):
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
        return get_mmlu_dataset(**kwargs)
    elif task.startswith('codex'):
        return get_codex_dataset(**kwargs)
    elif task.startswith('gsm'):
        return get_gsm_dataset(**kwargs)
    else:
        raise ValueError("Invalid task name")

