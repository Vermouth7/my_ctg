import json
import logging
import os

os.environ['CUDA_VISIBLE_DEVICES']='0'
import json
import random
import re

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from utils import prompt_template
from vllm import LLM, SamplingParams

random.seed(42)


template='''
As a linguist proficient in question-and-answer reasoning, the content of your task is to think about how you can better answer the question, uncover the detailed reasoning steps of the question and the constraints that were not explicitly stated in the original question, and output this additional information in the form of two instructions that the model can understand. I will provide you with the original question. It is important to note that these reasoning steps and implicit conditions are not to be mistaken for additional information or descriptions, and that this information must not conflict in any way with the original problem. Your response must include two new instructions.

Here are some examples:

Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
Reasoning steps:
  "instruction 1": "Calculate the number of clips sold in May by dividing the number of clips sold in April by 2."
  "instruction 2": "Add the number of clips sold in April and the number of clips sold in May to find the total number of clips sold."

Question: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
Reasoning steps:
  "instruction 1": "Convert the time Weng spent babysitting from minutes to hours by dividing 50 minutes by 60 to find the fraction of an hour worked.",
  "instruction 2": "Multiply the hourly rate of $12 by the fraction of an hour worked to calculate Weng's total earnings."

Question: %s

'''


with open('/home/chh/repos/moe_ctg/dataset/con_template_mmlu.txt','r',encoding='utf-8') as f:
    template=f.read()

ds = load_dataset("/data1/chh/datasets/lighteval/mmlu",'abstract_algebra')['test']

new_data=[]
num=0
sampling_params = SamplingParams(temperature=0.6, top_p=0.9,max_tokens=256)
model=LLM(model='/data1/chh/models/meta-llama/Meta-Llama-3-8B-Instruct',gpu_memory_utilization=0.90)
tokenizer = AutoTokenizer.from_pretrained('/data1/chh/models/meta-llama/Meta-Llama-3-8B-Instruct')

input_string=[]
for i in ds:
    input_string.append(template%(i['question']))
outputs=model.generate(input_string,sampling_params)
for d,output in zip(ds,outputs):
    content = output.outputs[0].text
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
    print(answer)
    
    answer={}
    answer['question']=d['question']
    answer['answer']=d['answer']
    answer['choices']=d['choices']
    answer['subject']=d['subject']
    answer['split']=content
    new_data.append(answer)
    
with open("./instructions/mmlu/mmlu_2steps_llama.json", "w", encoding="utf-8") as f:
    json.dump(new_data, f, ensure_ascii=False, indent=4)

print(f"save {len(new_data)} records")



with open('./instructions/mmlu/mmlu_2steps_llama.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

instruction1_pattern = r'"instruction 1":\s?"(.*?)",?\n'
instruction2_pattern = r'"instruction 2":\s?"(.*?)",?\n'

for item in data:
    split_content = item.get('split', '')
    
    instruction1_match = re.search(instruction1_pattern, split_content,re.DOTALL)
    instruction2_match = re.search(instruction2_pattern, split_content,re.DOTALL)
    if instruction1_match:
        item['instruction 1'] = instruction1_match.group(1).strip()
    
    if instruction2_match:
        item['instruction 2'] = instruction2_match.group(1).strip()
    del item['split']

with open('./instructions/mmlu/mmlu_2steps_llama.json', 'w', encoding='utf-8') as output_file:
    json.dump(data, output_file, ensure_ascii=False, indent=4)

print("Data processed and saved successfully!")
