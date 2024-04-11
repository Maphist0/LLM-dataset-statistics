"""
This script plots the request-response length distribution for the ShareGPT dataset.

Reference:
    https://huggingface.co/datasets/RyokoAI/ShareGPT52K
    https://huggingface.co/datasets/liyucheng/ShareGPT90K
"""
import numpy as np
from datasets import load_dataset
from transformers import LlamaTokenizer, GPT2Tokenizer

from utils.draw_utils import draw_req_res_distribution

dataset = "liyucheng/ShareGPT90K"
result = "img/sharegpt-req-res.png"

# in case it takes too long (about 40min) to process the full dataset
# randomly pick n samples from the train set.
nsamples = 2000 # set -1 to process the full dataset

# may use llama tokenizer or gpt2 tokenizer
# base_model = "baffo32/decapoda-research-llama-7B-hf"
# tokenizer = LlamaTokenizer.from_pretrained(base_model)
base_model = "openai-community/gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(base_model)

req_tokens = []
res_tokens = []
ill_format_cases = [] 

def generate_and_tokenize_prompt(data_point):
    """
    Input data_point is one sample from the dataset.
    This function returns nothing.
    """
    # early stop in case it takes too long to process the full dataset
    if nsamples > 0 and len(req_tokens) >= nsamples:
        return

    # preprocess and clean the sample
    conv = data_point['conversations']
    conv_from, conv_val = conv['from'], conv['value']
    if 'system' in conv['from']:
        idx = [i for i, e in enumerate(conv_from) if e == 'system']
        conv_from = [e for i, e in enumerate(conv_from) if i not in idx]
        conv_val = [e for i, e in enumerate(conv_val) if i not in idx]
    if 'chatgpt' in conv['from']:
        conv_from = ['gpt' if e == 'chatgpt' else e for i, e in enumerate(conv_from)]
    if 'bing' in conv['from']:
        conv_from = ['gpt' if e == 'bing' else e for i, e in enumerate(conv_from)]
    if 'bard' in conv['from']:
        conv_from = ['gpt' if e == 'bard' else e for i, e in enumerate(conv_from)]
    if 'user' in conv['from']:
        conv_from = ['human' if e == 'user' else e for i, e in enumerate(conv_from)]
    if 'assistant' in conv['from']:
        conv_from = ['gpt' if e == 'assistant' else e for i, e in enumerate(conv_from)]
    
    # in cases like: [human, gpt, human, human, human, gpt],
    # merge consecutive human (or gpt) sequences into one entry
    if len(conv_from) > 1:
        i = 1
        while i < len(conv_from):
            if conv_from[i] == conv_from[i-1]:
                conv_from.pop(i)
                conv_val[i-1] = conv_val[i-1] + conv_val.pop(i)
            else:
                i += 1

    # if the first seq is from gpt, ignore it
    if len(conv_from) > 1 and conv_from[0] == 'gpt':
        conv_from.pop(0)
        conv_val.pop(0)

    # if only has one human prompt, skip
    if len(conv_from) == 1:
        return

    curr_req_tokens = []
    curr_res_tokens = []
    for i in range(len(conv_from)):
        if i % 2 == 0:
            if conv_from[i] != 'human':
                if len(conv_from) > 1:
                    print(f"incorrect sequence of 'from': {conv_from}")
                ill_format_cases.append(conv)
                return
        else:
            if conv_from[i] != 'gpt':
                if len(conv_from) > 1:
                    print(f"incorrect sequence of 'from': {conv_from}")
                ill_format_cases.append(conv)
                return

        result = tokenizer(
            conv_val[i],
            truncation=True,
            max_length=4096,
            padding=False,
            return_tensors=None,
        )

        if i % 2 == 0: # human
            curr_req_tokens.append(len(result['input_ids']))
        else: # gpt
            curr_res_tokens.append(len(result['input_ids']))

    # if the last req is from human, ignore it
    if len(curr_req_tokens) > len(curr_res_tokens):
        curr_req_tokens.pop(-1)

    # only save its result if successful
    req_tokens.extend(curr_req_tokens)
    res_tokens.extend(curr_res_tokens)

ds = load_dataset(dataset)
train_data = ds["train"].shuffle().map(generate_and_tokenize_prompt)

assert len(req_tokens) == len(res_tokens)
print(f"{len(req_tokens)} conversation samples")

draw_req_res_distribution(
    np.array(req_tokens), np.array(res_tokens), result,
    xlabel='Request Tokens - GPT in ShareGPT (Conv.)')