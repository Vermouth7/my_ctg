import json
import logging
import os
import random

import regex as re

os.environ['CUDA_VISIBLE_DEVICES']='7'
from datasets import load_dataset
from human_eval.data import read_problems, write_jsonl
from openai import OpenAI
from tqdm import tqdm
from transformers import AutoTokenizer
from utils import prompt_template
from vllm import LLM, SamplingParams

random.seed(42)


sampling_params = SamplingParams(temperature=0.6, top_p=0.9,max_tokens=256)
model=LLM(model='/data1/chh/models/meta-llama/Meta-Llama-3-8B-Instruct',gpu_memory_utilization=0.90)
tokenizer = AutoTokenizer.from_pretrained('/data1/chh/models/meta-llama/Meta-Llama-3-8B-Instruct')
template=None
with open('/home/chh/repos/my_ctg/instructions/template/con_template_humaneval.txt','r',encoding='utf-8') as f:
    template=f.read()

ds = read_problems()


new_data=[]    
input_string=[]
for task_id in ds:
    i=ds[task_id]
    input_string.append(template%(i['prompt']))
outputs=model.generate(input_string,sampling_params)
for task_id,output in zip(ds,outputs):
    content = output.outputs[0].text
    d=ds[task_id]
    answer={}
    answer['task_id']=d['task_id']
    answer['prompt']=d['prompt']
    answer['canonical_solution']=d['canonical_solution']
    answer['split']=content
    new_data.append(answer)


for item in new_data:
    split_content = item.get('split', '')
    split_content.strip()
    json_match = re.search(r'Instructions:\s*(\{(?:[^{}]|(?1))*\})', split_content, re.DOTALL)

    if json_match:
        json_text = json_match.group(1)
        
        try:
            json_data = json.loads(json_text)
            if json_data:
                item['instruction 1'] = json_data['instruction 1']
                item['instruction 2'] = json_data['instruction 2']
                del item['split']
        except json.JSONDecodeError:
            pattern_1 = r'"instruction 1":\s*"(.*?)"'
            pattern_2 = r'"instruction 2":\s*"(.*?)"'
            
            instruction_1 = re.search(pattern_1, item['split'])
            instruction_2 = re.search(pattern_2, item['split'])
            if instruction_1 and instruction_2:
                item['instruction 1']=instruction_1.group(1)[0]
                item['instruction 2']=instruction_2.group(1) 
            del item['split']
            
            # print(item)

with open('./instructions/humaneval/humaneval_2steps_llama.json', 'w', encoding='utf-8') as output_file:
    json.dump(new_data, output_file, ensure_ascii=False, indent=4)

print("Data processed and saved successfully!")

    
# with open('./instructions/humaneval/humaneval_2steps_llama.json', 'r', encoding='utf-8') as output_file:
#     data=json.load(output_file)
# openai_api_key = "ak-Lfu504S5OrzjNvivYYdY6E8xvn1hiTY42texx7WvTIojB9MC"
# openai_api_base = "https://api.nextapi.fun"
# for item in tqdm(data):
#     if ('instruction 1' not in item) or ('instruction 2' not in item):
#         ins=template%(item['problem'])
#         client = OpenAI(base_url=openai_api_base,api_key=openai_api_key)
#         response = client.chat.completions.create(
#             model='gpt-4o-mini',
#             messages=[{
#                 'role': 'user',
#                 'content': ins,
#             }],
#         )
#         content = response.choices[0].message.content
#         answer=None
#         try:
#             if content.startswith("```json"): # remove markdown, used for gpt-4 turbo
#                 content = content[7:-3].strip()
#                 answer = json.loads(content)
#             else:
#                 answer = json.loads(content)
#         except Exception as e:
#                 print(f"json failed to parse: {e}")
#                 print(f"content: {content}")
#         if answer:
#             item['instruction 1']=answer['key point 1']
#             item['instruction 2']=answer['key point 2']
            
# with open('./instructions/humaneval/humaneval_2steps_llama.json', 'w', encoding='utf-8') as output_file:
#     json.dump(data, output_file, ensure_ascii=False, indent=4)