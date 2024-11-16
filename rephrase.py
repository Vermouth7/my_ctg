import json
import logging
import os
import random
import re

os.environ['CUDA_VISIBLE_DEVICES']='3'
from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm
from transformers import AutoTokenizer
from utils import prompt_template
from vllm import LLM, SamplingParams

random.seed(42)


# sampling_params = SamplingParams(max_tokens=256)
# model=LLM(model='/data1/chh/models/meta-llama/Meta-Llama-3-8B-Instruct',gpu_memory_utilization=0.90)
# tokenizer = AutoTokenizer.from_pretrained('/data1/chh/models/meta-llama/Meta-Llama-3-8B-Instruct')
template=None
with open('/home/chh/repos/my_ctg/instructions/template/rephrase.txt','r',encoding='utf-8') as f:
    template=f.read()
with open('/home/chh/repos/my_ctg/instructions/ifeval/ifeval_2steps_llama_2.json','r',encoding='utf-8') as f:
    ins=json.load(f)

ds = load_dataset("/data1/chh/datasets/google/IFEval")
ds=ds['train']
    
new_data=[]    
input_string=[]
# for i in ins:
#     input_string.append(prompt_template(tokenizer,template%(i['prompt'],i['instruction 1'],i['instruction 2'])))
# outputs=model.generate(input_string,sampling_params)
# for d,output in zip(ins,outputs):
#     content = output.outputs[0].text
#     answer={}
#     answer['prompt']=d['prompt']
#     answer['instruction 1']=d['instruction 1']
#     answer['instruction 2']=d['instruction 2']
#     answer['new_prompt']=content
#     new_data.append(answer)

openai_api_key = "ak-Lfu504S5OrzjNvivYYdY6E8xvn1hiTY42texx7WvTIojB9MC"
openai_api_base = "https://api.nextapi.fun"
for item in tqdm(ins):
    meg=template%(item['prompt'],item['instruction 1'],item['instruction 2'])
    client = OpenAI(base_url=openai_api_base,api_key=openai_api_key)
    response = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[{
            'role': 'user',
            'content': meg,
        }],
    )
    content = response.choices[0].message.content
    print(content)
    answer={}
    answer['prompt']=item['prompt']
    answer['instruction 1']=item['instruction 1']
    answer['instruction 2']=item['instruction 2']
    answer['new_prompt']=content
    new_data.append(answer)
            

with open('./instructions/ifeval/ifeval_rephrase.json', 'w', encoding='utf-8') as output_file:
    json.dump(new_data, output_file, ensure_ascii=False, indent=4)

print("Data processed and saved successfully!")