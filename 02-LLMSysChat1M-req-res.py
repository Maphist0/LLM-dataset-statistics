"""
This script plots the request-response length distribution for the LLMSys-Chat-1M dataset.

Reference:
    Lianmin Zheng, et al., LMSYS-Chat-1M: A Large-Scale Real-World LLM Conversation Dataset
    https://huggingface.co/datasets/lmsys/lmsys-chat-1m
"""
import numpy as np
from datasets import load_dataset
from transformers import LlamaTokenizer, GPT2Tokenizer

from utils.draw_utils import draw_req_res_distribution

dataset = "lmsys/lmsys-chat-1m"
result = "img/llmsyschat1m-req-res.png"

# in case it takes too long to process the full dataset
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
    conv = data_point['conversation']
    conv_from = [i['role'] for i in conv]
    conv_val  = [i['content'] for i in conv]

    # if only has one human prompt, skip
    if len(conv_from) == 1:
        return

    curr_req_tokens = []
    curr_res_tokens = []
    for i in range(len(conv_from)):
        if i % 2 == 0:
            if conv_from[i] != 'user':
                print(f"incorrect sequence of 'from': {conv_from}")
                ill_format_cases.append(conv)
                return
        else:
            if conv_from[i] != 'assistant':
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

        if i % 2 == 0: # user
            curr_req_tokens.append(len(result['input_ids']))
        else: # assistant
            curr_res_tokens.append(len(result['input_ids']))

    # if the last req is from user, ignore it
    if len(curr_req_tokens) > len(curr_res_tokens):
        curr_req_tokens.pop(-1)

    # only save its result if successful
    req_tokens.extend(curr_req_tokens)
    res_tokens.extend(curr_res_tokens)

ds = load_dataset(dataset)
train_data = ds["train"].shuffle().map(generate_and_tokenize_prompt)

assert len(req_tokens) == len(res_tokens)
print(f"{len(req_tokens)} conversation samples")

draw_req_res_distribution(np.array(req_tokens), np.array(res_tokens), result,
                          xlabel="Request Tokens - LLMSys-Chat-1M (Conv.)")