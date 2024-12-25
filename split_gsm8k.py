import json
import logging
import os
import random
import re

os.environ['CUDA_VISIBLE_DEVICES']='4'
import torch
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
def extract_bracket_content(text):
    """
    Extracts the content inside square brackets [] from the input text
    and converts it into a Python list.

    Args:
        text (str): The input text containing square brackets and content.

    Returns:
        list: A Python list containing the extracted items.
    """
    # Use regex to extract the content inside the square brackets
    match = re.search(r'\[(.*?)\]', text, re.DOTALL)
    if not match:
        return []  # Return an empty list if no brackets are found

    # Extract content and split into individual items
    content = match.group(1)
    items = [item.strip().strip('"').strip("'") for item in content.split(",") if item.strip()]
    return items
# sampling_params = SamplingParams(max_tokens=1024)
# model=LLM(model='/data1/chh/models/meta-llama/Meta-Llama-3-8B-Instruct',gpu_memory_utilization=0.90)
# tokenizer = AutoTokenizer.from_pretrained('/data1/chh/models/meta-llama/Meta-Llama-3-8B-Instruct')
# template=None
with open('/home/chh/repos/my_ctg/instructions/template/con_template_math.txt','r',encoding='utf-8') as f:
    template=f.read()

# # with open('/home/chh/repos/my_ctg/results/gsm8k_act/train_correct.json','r',encoding='utf-8') as f:
# #     ds=json.load(f)
# # ds = load_dataset("/data1/chh/datasets/openai/gsm8k",'main')
# # ds=ds['test']
# ds=load_dataset("/data1/chh/datasets/lighteval/MATH")
# ds=ds['test']

# new_data=[]    
# input_string=[]
# for i in ds:
#     input_string.append(prompt_template(tokenizer,template%(i['problem'])))
# outputs=model.generate(input_string,sampling_params)
# for d,output in zip(ds,outputs):
#     content = output.outputs[0].text
#     answer={}
#     answer['problem']=d['problem']
#     answer['solution']=d['solution']
#     answer['split']=content
#     new_data.append(answer)
new_data_2=[]

with open('/home/chh/repos/my_ctg/instructions/math/math_message_3.json','r',encoding='utf-8') as f:
    new_data=json.load(f)
# for item in new_data:
#     if 'split' in item.keys():
#         split_content = item.get('split', '')
#         split_content=split_content.strip()
        # pattern = r"\{.*?\}"
        # matches = re.search(pattern, split_content, re.DOTALL)
        # if matches:
        #     json_text = matches.group(0)  
        #     fixed_json_text = fix_json_format(json_text)
        #     try:
        #         data_dict = json.loads(json_text)
        #         # if 'instruction 1' in data_dict.keys():
        #         #     item['instruction 1']=data_dict['instruction 1']
        #         # if 'instruction 2' in data_dict.keys():
        #         #     item['instruction 2']=data_dict['instruction 2']
        #         #     del item['split']
        #         data_dict['problem']=item['problem']
        #         data_dict['solution']=item['solution']
        #         new_data_2.append(data_dict)
        #     except json.JSONDecodeError as e:
        #         # data_dict=extract_instructions(split_content)
        #         # if 'instruction 1' in data_dict.keys():
        #         #     item['instruction 1']=data_dict['instruction 1']
        #         # if 'instruction 2' in data_dict.keys():
        #         #     item['instruction 2']=data_dict['instruction 2']
        #         #     del item['split']
        #         new_data_2.append(item)
        #         print(json_text)
        # else:
        #     new_data_2.append(item)
        # # else:
        # #     data_dict=extract_instructions(split_content)
        # #     if 'instruction 1' in data_dict.keys():
        # #         item['instruction 1']=data_dict['instruction 1']
        # #     if 'instruction 2' in data_dict.keys():
        # #         item['instruction 2']=data_dict['instruction 2']
        # #         del item['split']
#         result = extract_bracket_content(split_content)
#         if result:
#             data_dict={}
#             data_dict['list']=result
#             data_dict['problem']=item['problem']
#             data_dict['solution']=item['solution']
#             new_data_2.append(data_dict)
#         else:
#             new_data_2.append(item)
#     else:
#         new_data_2.append(item)

# with open('./instructions/math/math_message_3.json', 'w', encoding='utf-8') as output_file:
#     json.dump(new_data_2, output_file, ensure_ascii=False, indent=4)

# print("Data processed and saved successfully!")

    

openai_api_key = "ak-Lfu504S5OrzjNvivYYdY6E8xvn1hiTY42texx7WvTIojB9MC"
openai_api_base = "https://api.nextapi.fun"
for item in tqdm(new_data):
    if 'split' in item.keys():
        ins=template%(item['problem'])
        client = OpenAI(base_url=openai_api_base,api_key=openai_api_key)
        response = client.chat.completions.create(
            model='gpt-4o-mini',
            messages=[{
                'role': 'user',
                'content': ins,
            }],
        )
        content = response.choices[0].message.content
        result = extract_bracket_content(content)
        if result:
            data_dict={}
            data_dict['list']=result
            data_dict['problem']=item['problem']
            data_dict['solution']=item['solution']
            new_data_2.append(data_dict)
        else:
            new_data_2.append(item)
    else:
        new_data_2.append(item)
        # answer=None
        # try:
        #     if content.startswith("```json"): # remove markdown, used for gpt-4 turbo
        #         content = content[7:-3].strip()
        #         answer = json.loads(content)
        #     else:
        #         answer = json.loads(content)
        # except Exception as e:
        #         print(f"json failed to parse: {e}")
        #         print(f"content: {content}")
        # if answer:
        #     item['instruction 1']=answer['instruction 1']
        #     item['instruction 2']=answer['instruction 2']
            
with open('./instructions/gsm8k/gsm8k_2steps_llama_4.json', 'w', encoding='utf-8') as output_file:
    json.dump(new_data_2, output_file, ensure_ascii=False, indent=4)