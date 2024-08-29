import os

os.environ['CUDA_VISIBLE_DEVICES']='7'
import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F
from batch_repe import repe_pipeline_registry
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, LlamaForCausalLM, pipeline
from utils import *
from vllm import LLM, SamplingParams

device = torch.device("cuda")
constraint_types=['content', 'situation', 'style', 'format', 'example', 'mixed']
data_path='/home/chh/repos/FollowBench/data/'
api_input_path='/home/chh/repos/my_ctg/instructions/followbench/'
instruction_template = """
Break down the following complex instruction or question into {num} simpler instructions, each containing only a single constraint. 
Ensure that each new instruction does not contradict the original instruction's meaning or logic, and remains meaningful. 
For each new instruction, brainstorm ways to make it clearer and identify any implicit constraints that might be present.
Number each simpler instruction sequentially. Your response must only contain simpler instructions without any any other descriptive statements.
Please surround all simple commands with '[ ]', and be sure to follow the format of the response: 1. [ ]\n2. [ ]\n3.[ ]
Instruction or question: {complex_instruction}
"""
for constraint_type in constraint_types:
    convert_to_api_input(
                    data_path=data_path, 
                    api_input_path=api_input_path, 
                    constraint_type=constraint_type
                    )

model_path='/data1/chh/models/meta-llama/Meta-Llama-3-8B-Instruct'
model = LLM(model=model_path,gpu_memory_utilization=0.95)
tokenizer = AutoTokenizer.from_pretrained(model_path)
sampling_params = SamplingParams(temperature=0.6, top_p=0.9,max_tokens=500)


for constraint_type in constraint_types:
    data = []
    prompts=[]
    
    
    with open(os.path.join(api_input_path, f"{constraint_type}_constraint.jsonl"), 'r', encoding='utf-8') as data_file:
        for line in data_file:
            data.append(json.loads(line))

    for i in range(len(data)):
        ins=prompt_template(tokenizer,instruction_template.format(num=data[i]['level']+1,complex_instruction=data[i]['prompt_new']),'You are a helpful assistant!')
        
        prompts.append(ins)



    outputs = model.generate(prompts, sampling_params)
    res=[]
    for item,output in zip(data,outputs):
        generated_text = output.outputs[0].text
        tmp = {'original_ins':item['prompt_new'],'split_ins':generated_text,'level':item['level']}
        res.append(tmp)
    with open(os.path.join(api_input_path, "{}_constraint_split.jsonl".format(constraint_type)), 'w', encoding='utf-8') as output_file:
        for d in res:
            if d['level'] > 0:
                output_file.write(json.dumps({'original_ins': d['original_ins'],'split_ins':d['split_ins'],'level':d['level']})+ "\n")
        