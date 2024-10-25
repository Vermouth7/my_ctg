import os

os.environ['CUDA_VISIBLE_DEVICES']='4'
import argparse
import json
import time

import numpy as np
import torch
from openai import OpenAI
from tqdm import tqdm
from utils import *
from vllm import LLM, SamplingParams

device = torch.device("cuda")
constraint_types=['format', 'example', 'mixed']

data_path='/home/chh/repos/FollowBench/data/'
api_input_path='/home/chh/repos/my_ctg/instructions/followbench5/'


for constraint_type in constraint_types:
    convert_to_api_input(
                    data_path=data_path, 
                    api_input_path=api_input_path, 
                    constraint_type=constraint_type
                    )

# sampling_params = SamplingParams(temperature=0.6, top_p=0.9,max_tokens=1024)
# model=LLM(model='/data1/chh/models/meta-llama/Meta-Llama-3-8B-Instruct',gpu_memory_utilization=0.90)
# tokenizer = AutoTokenizer.from_pretrained('/data1/chh/models/meta-llama/Meta-Llama-3-8B-Instruct')
template=None
with open('/home/chh/repos/my_ctg/instructions/template/con_template_followbench.txt','r',encoding='utf-8') as f:
    template=f.read()

stream = False

for constraint_type in constraint_types:
    data = []
    prompts=[]
    new_data=[]
    
    res=[]
    # with open(os.path.join(api_input_path, f"{constraint_type}_constraint.jsonl"), 'r', encoding='utf-8') as data_file:
    #     for line in data_file:
    #         data.append(json.loads(line))
    # input_string=[]
    # for i in data:
        # input_string.append(prompt_template(tokenizer,template%(i['prompt_new'])))
        # input_string.append(template%(i['prompt_new']))
        
    # outputs=model.generate(input_string,sampling_params)
    # for d,output in zip(data,outputs):
    #     content = output.outputs[0].text
    #     answer={}
    #     answer['prompt_new']=d['prompt_new']
    #     answer['level']=d['level']
    #     answer['split']=content
    #     new_data.append(answer)


    # for item in new_data:
    #     split_content = item.get('split', '')
    #     split_content.strip()
    #     json_match = re.search(r'({.*?})', split_content, re.DOTALL)

    #     if json_match:
    #         json_text = json_match.group(1)
            
    #         try:
    #             json_data = json.loads(json_text)
    #             if json_data and 'instruction 1' in json_data.keys() and 'instruction 2' in json_data.keys():
    #                 item['instruction 1'] = json_data['instruction 1']
    #                 item['instruction 2'] = json_data['instruction 2']
    #                 del item['split']
    #         except json.JSONDecodeError:
    #             pattern_1 = r'"instruction 1":\s*"(.*?)"'
    #             pattern_2 = r'"instruction 2":\s*"(.*?)"'
                
    #             instruction_1 = re.search(pattern_1, item['split'])
    #             instruction_2 = re.search(pattern_2, item['split'])
    #             if instruction_1 and instruction_2:
    #                 item['instruction 1']=instruction_1.group(1)
    #                 item['instruction 2']=instruction_2.group(1) 
    #                 del item['split']
                
    #             # print(item)

    # with open(os.path.join(api_input_path, "{}_constraint_split.jsonl".format(constraint_type)), 'w', encoding='utf-8') as output_file:
    #     for d in new_data:
    #         if d['level'] > 0:
    #             output_file.write(json.dumps(d)+ "\n")
    # print("Data processed and saved successfully!")
    
    
    with open(os.path.join(api_input_path, f"{constraint_type}_constraint.jsonl"), 'r', encoding='utf-8') as data_file:
        for line in data_file:
            data.append(json.loads(line))
    openai_api_key = "ak-Lfu504S5OrzjNvivYYdY6E8xvn1hiTY42texx7WvTIojB9MC"
    openai_api_base = "https://api.nextapi.fun"
    for item in tqdm(data):
        # if ('instruction 1' not in item) or ('instruction 2' not in item):
        ins=template%(item['prompt_new'])
        client = OpenAI(base_url=openai_api_base,api_key=openai_api_key)
        response = client.chat.completions.create(
            model='gpt-4o-mini',
            messages=[{
                'role': 'user',
                'content': ins,
            }],
        )
        if response:
            content = response.choices[0].message.content
        answer=None
        try:
            if content.startswith("```json"): # remove markdown, used for gpt-4 turbo
                content = content[7:-3].strip()
                answer = json.loads(content)
            else:
                answer = json.loads(content)
        except Exception as e:
                print(f"json failed to parse: {e}")
                print(f"content: {content}")
        if isinstance(answer,dict):
            for key in answer.keys():
                item[key]=answer[key]
        else:
            item['split']=content
    with open(os.path.join(api_input_path, "{}_constraint_split.jsonl".format(constraint_type)), 'w', encoding='utf-8') as output_file:
        for d in data:
            if d['level'] > 0:
                output_file.write(json.dumps(d)+ "\n")         