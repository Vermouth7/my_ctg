import json
import logging
import os
import random
import re

os.environ['CUDA_VISIBLE_DEVICES']='7'
from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm
from transformers import AutoTokenizer
from utils import prompt_template
from vllm import LLM, SamplingParams

random.seed(42)


with open('/home/chh/repos/my_ctg/instructions/gsm8k/gsm8k_2steps_llama_5_label.json', 'r', encoding='utf-8') as output_file:
    data=json.load(output_file)

"""
<THOUGHT> something as self-reminder.<\THOUGHT>
"""
for item in tqdm(data):
    item['instruction 1']='<THOUGHT> '+item['instruction 1']+' <\THOUGHT>'
    item['instruction 2']='<THOUGHT> '+item['instruction 2']+' <\THOUGHT>'
            
with open('/home/chh/repos/my_ctg/instructions/gsm8k/gsm8k_2steps_llama_5_label.json', 'w', encoding='utf-8') as output_file:
    json.dump(data, output_file, ensure_ascii=False, indent=4)