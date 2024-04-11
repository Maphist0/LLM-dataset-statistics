"""
This script reproduces Figure 8 in BurstGPT paper.

Reference:
    Yuxin Wang, et al., Towards Efficient and Reliable LLM Serving: A Real-World Workload Study

Prepare data:
    Download "BurstGPT_without_fails.csv" from the official github repo.
    https://github.com/HPMLL/BurstGPT
"""
import numpy as np

from utils.draw_utils import draw_req_res_distribution

fname = "BurstGPT_without_fails.csv"
result = "img/burstgpt-fig8.png"

with open(fname, 'r') as ifile:
    lines = ifile.readlines()

lines = [l for l in lines[1:] if "Conversation" in l.split(',')[-1] and 'ChatGPT' in l.split(',')[1]]
print(f"{len(lines)} ChatGPT conversation samples")

req_tokens = np.array([int(l.split(',')[2]) for l in lines])
res_tokens = np.array([int(l.split(',')[3]) for l in lines])

draw_req_res_distribution(req_tokens, res_tokens, result, 
                          xlabel='Request Tokens - ChatGPT (Conv.)')