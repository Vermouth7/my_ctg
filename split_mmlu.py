import json
import logging
import os

os.environ['CUDA_VISIBLE_DEVICES']='0'
import json
import random
import re

import pandas as pd
from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm
from transformers import AutoTokenizer
from utils import prompt_template
from vllm import LLM, SamplingParams

random.seed(42)

with open('/home/chh/repos/my_ctg/instructions/template/con_template_mmlu.txt','r',encoding='utf-8') as f:
    template=f.read()
subjects = sorted(
        [
            f.split("_test.csv")[0]
            for f in os.listdir(os.path.join('/data1/chh/datasets/lighteval/data', "test"))
            if "_test.csv" in f
        ]
    )
openai_api_key = "ak-Lfu504S5OrzjNvivYYdY6E8xvn1hiTY42texx7WvTIojB9MC"
openai_api_base = "https://api.nextapi.fun"

# sampling_params = SamplingParams(temperature=0.6, top_p=0.9,max_tokens=256)
# model=LLM(model='/data1/chh/models/meta-llama/Meta-Llama-3-8B-Instruct',gpu_memory_utilization=0.90)
# tokenizer = AutoTokenizer.from_pretrained('/data1/chh/models/meta-llama/Meta-Llama-3-8B-Instruct')

for subject in subjects:

    ds = load_dataset("/data1/chh/datasets/lighteval/mmlu",subject)['test']

    # new_data=[]    
    # input_string=[]
    # for i in ds:
    #     input_string.append(template%(i['question']))
    # outputs=model.generate(input_string,sampling_params)
    
    # for d,output in zip(ds,outputs):
    #     content = output.outputs[0].text
    #     answer=None
    #     try:
    #         if content.startswith("```json"): # remove markdown, used for gpt-4 turbo
    #             content = content[7:-3].strip()
    #             answer = json.loads(content)
    #         else:
    #             answer = json.loads(content)
    #     except Exception as e:
    #             # print(f"json failed to parse: {e}")
    #             # print(f"content: {content}")
    #             pass
        
    #     answer={}
    #     answer['question']=d['question']
    #     answer['answer']=d['answer']
    #     answer['choices']=d['choices']
    #     answer['subject']=d['subject']
    #     answer['split']=content
    #     new_data.append(answer)
        
    # with open("./instructions/mmlu/{}_2steps_llama.json".format(subject), "w", encoding="utf-8") as f:
    #     json.dump(new_data, f, ensure_ascii=False, indent=4)

    # print(f"save {len(new_data)} records")



    # with open('./instructions/mmlu/{}_2steps_llama.json'.format(subject), 'r', encoding='utf-8') as file:
    #     data = json.load(file)

    # instruction1_pattern = r'"instruction 1":\s?"(.*?)",?\n'
    # instruction2_pattern = r'"instruction 2":\s?"(.*?)",?\n'

    # for item in data:
    #     split_content = item.get('split', '')
        
    #     instruction1_match = re.search(instruction1_pattern, split_content,re.DOTALL)
    #     instruction2_match = re.search(instruction2_pattern, split_content,re.DOTALL)
    #     if instruction1_match:
    #         item['instruction 1'] = instruction1_match.group(1).strip()
        
    #     if instruction2_match:
    #         item['instruction 2'] = instruction2_match.group(1).strip()
    #     else:
    #         print(item['question'])
    #     del item['split']

    # with open('./instructions/mmlu/{}_2steps_llama.json'.format(subject), 'w', encoding='utf-8') as output_file:
    #     json.dump(data, output_file, ensure_ascii=False, indent=4)

    # print("Data processed and saved successfully!")
    
    with open('./instructions/mmlu_gpt/{}_2steps_gpt.json'.format(subject), 'r', encoding='utf-8') as file:
        data = json.load(file)

    
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
                
    # with open('./instructions/mmlu/{}_2steps_llama.json'.format(subject), 'w', encoding='utf-8') as output_file:
    #     json.dump(data, output_file, ensure_ascii=False, indent=4)