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
def fix_json_format(json_text):
    """
    自动修复 JSON 文本中可能缺失的逗号问题
    """
    lines = json_text.splitlines()
    fixed_lines = []
    
    for i in range(len(lines) - 1):
        current_line = lines[i].strip()
        next_line = lines[i + 1].strip()
        
        # 如果当前行以引号结尾，下一行不是 '}' 或 ']'，则补充逗号
        if current_line.endswith('"') and not (next_line.startswith('}') or next_line.startswith(']')):
            current_line += ','
        
        fixed_lines.append(current_line)
    
    # 添加最后一行
    fixed_lines.append(lines[-1].strip())
    return "\n".join(fixed_lines)
def extract_instructions(text):
    """
    从文本中提取 instruction 格式部分，并构造字典
    """
    # 匹配 instruction 的模式
    pattern = r'"(instruction \d+)":\s*"(.*?)"'
    matches = re.findall(pattern, text, re.DOTALL)
    
    # 构造字典
    
    instructions_dict = {}
    for match in matches:
        key = match[0].strip(":")  # 提取键名
        value = match[1].strip()  # 提取值，去除首尾多余空白
        instructions_dict[key] = value
        
    return instructions_dict
# sampling_params = SamplingParams(max_tokens=1024)
# model=LLM(model='/data1/chh/models/meta-llama/Meta-Llama-3-8B-Instruct',gpu_memory_utilization=0.90)
# tokenizer = AutoTokenizer.from_pretrained('/data1/chh/models/meta-llama/Meta-Llama-3-8B-Instruct')
# template=None
with open('/home/chh/repos/my_ctg/instructions/template/key_mes.txt','r',encoding='utf-8') as f:
    template=f.read()

# with open('/home/chh/repos/my_ctg/results/gsm8k_act/train_correct.json','r',encoding='utf-8') as f:
#     ds=json.load(f)
# ds = load_dataset("/data1/chh/datasets/openai/gsm8k",'main')
# ds=ds['test']

# new_data=[]    
# input_string=[]
# for i in ds:
#     input_string.append(prompt_template(tokenizer,template%(i['question'])))
# outputs=model.generate(input_string,sampling_params)
# for d,output in zip(ds,outputs):
#     content = output.outputs[0].text
#     answer={}
#     answer['question']=d['question']
#     answer['answer']=d['answer']
#     answer['split']=content
#     new_data.append(answer)

with open('./instructions/gsm8k/gsm8k_message.json', 'r', encoding='utf-8') as output_file:
    new_data=json.load(output_file)
for item in new_data:
    # if 'split' in item.keys():
    #     split_content = item.get('split', '')
    #     split_content=split_content.strip()
    #     # pattern = r"\{.*?\}"
    #     # matches = re.search(pattern, split_content, re.DOTALL)
    #     # if matches:
    #     #     json_text = matches.group(0)  
    #     #     fixed_json_text = fix_json_format(json_text)
    #     #     try:
    #     #         data_dict = json.loads(json_text)
    #     #         for k,v in data_dict.items():
    #     #             item[k]=v
    #     #         del item['split']
    #     #     except json.JSONDecodeError as e:
    #     #         print(fixed_json_text)
    #     split_list=split_content.split('\n')
    #     temp=[]
    #     for i in range(0,len(split_list)):
    #         if len(split_list[i])>2 and i!=0:
    #             temp.append(split_list[i])
    #     item['list']=temp
    if 'list' in item.keys():
        for i in range(0,len(item['list'])):
            if i == len(item['list'])-1:
                if 'Note' in item['list'][i]:
                    item['list'].pop(i)


with open('./instructions/gsm8k/gsm8k_message.json', 'w', encoding='utf-8') as output_file:
    json.dump(new_data, output_file, ensure_ascii=False, indent=4)

print("Data processed and saved successfully!")

    

# openai_api_key = "ak-Lfu504S5OrzjNvivYYdY6E8xvn1hiTY42texx7WvTIojB9MC"
# openai_api_base = "https://api.nextapi.fun"
# for item in tqdm(data):
#     if ('instruction 1' not in item) or ('instruction 2' not in item):
#         ins=template%(item['question'])
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
#             item['instruction 1']=answer['instruction 1']
#             item['instruction 2']=answer['instruction 2']
            
# with open('./instructions/gsm8k/gsm8k_2steps_llama_2.json', 'w', encoding='utf-8') as output_file:
#     json.dump(data, output_file, ensure_ascii=False, indent=4)