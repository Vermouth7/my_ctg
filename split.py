import os

os.environ['CUDA_VISIBLE_DEVICES']='7'
import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F
from batch_repe import repe_pipeline_registry
from openai import OpenAI
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, LlamaForCausalLM, pipeline
from utils import *
from vllm import LLM, SamplingParams

device = torch.device("cuda")
constraint_types=['content', 'situation', 'style', 'format', 'example', 'mixed']
data_path='/home/chh/repos/FollowBench/data/'
api_input_path='/home/chh/repos/my_ctg/instructions/followbench3/'

# instruction_template = """
# Decompose the following complex instruction or problem into {num} simpler instructions, each containing only a single constraint. 
# Ensure that each new instruction does not contradict the original instruction's meaning or logic, and remains meaningful. 
# For each new instruction, brainstorm ways to make it clearer in conjunction with the original contextual information, and identify any implied limitations that may exist and how it could be better accomplished.
# Number each simpler instruction sequentially. Your response must only contain simpler instructions without any any other descriptive statements.
# Please surround all simple commands with '[ ]', and be sure to follow the format of the response: 1. [ ]\n2. [ ]\n
# Instruction or question: {complex_instruction}
# """

instruction_template2 = """
Decompose the following complex instruction or problem into 2 simpler instructions, each performing half the duties of the original instruction.
Ensure that each new instruction does not contradict the original instruction's meaning or logic, and remains meaningful. 
For each new instruction, brainstorm ways to make it clearer in conjunction with the original contextual information, and identify any implied limitations that may exist and how it could be better accomplished.
Number each simpler instruction sequentially. Your response must only contain simpler instructions without any any other descriptive statements.
Please surround all simple commands with '[ ]', and be sure to follow the format of the response: 1. [ ]\n2. [ ]
Instruction or question: {complex_instruction}
"""

for constraint_type in constraint_types:
    convert_to_api_input(
                    data_path=data_path, 
                    api_input_path=api_input_path, 
                    constraint_type=constraint_type
                    )

# model_path='/data1/chh/models/meta-llama/Meta-Llama-3-8B-Instruct'
# model = LLM(model=model_path,gpu_memory_utilization=0.95)
# tokenizer = AutoTokenizer.from_pretrained(model_path)
# sampling_params = SamplingParams(temperature=0.6, top_p=0.9,max_tokens=200)
openai_api_key = "ak-Lfu504S5OrzjNvivYYdY6E8xvn1hiTY42texx7WvTIojB9MC"
openai_api_base = "https://api.nextapi.fun"




stream = False



for constraint_type in constraint_types:
    data = []
    prompts=[]
    
    res=[]
    with open(os.path.join(api_input_path, f"{constraint_type}_constraint.jsonl"), 'r', encoding='utf-8') as data_file:
        for line in data_file:
            data.append(json.loads(line))

    for i in tqdm(range(len(data))):
        # ins=prompt_template(tokenizer,instruction_template.format(complex_instruction=data[i]['prompt_new']),'You are a helpful assistant!')
        ins=instruction_template2.format(complex_instruction=data[i]['prompt_new'])
        client = OpenAI(base_url=openai_api_base,api_key=openai_api_key)
        response = client.chat.completions.create(
            model='gpt-4o-mini',
            messages=[{
                'role': 'user',
                'content': ins,
            }],
        )
        content = response.choices[0].message.content
        res.append(content)
        print(content)
        # prompts.append(ins)
        # print("Completion results:")
    
        # print(content)



    # outputs = model.generate(prompts, sampling_params)
    # res=[]
    # for item,output in zip(data,outputs):
    #     generated_text = output.outputs[0].text
    #     tmp = {'original_ins':item['prompt_new'],'split_ins':generated_text,'level':item['level']}
    #     res.append(tmp)

    with open(os.path.join(api_input_path, "{}_constraint_split.jsonl".format(constraint_type)), 'w', encoding='utf-8') as output_file:
        for d,r in zip(data,res):
            if d['level'] > 0:
                output_file.write(json.dumps({'original_ins': d['prompt_new'],'split_ins':r,'level':d['level']})+ "\n")
        