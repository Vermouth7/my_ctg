import argparse
import json
import os

os.environ['CUDA_VISIBLE_DEVICES']='0'

import re
import time
from difflib import SequenceMatcher

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from batch_repe import repe_pipeline_registry
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import (AutoModel, AutoTokenizer, LlamaForCausalLM,
                          StoppingCriteria, StoppingCriteriaList, pipeline)
from utils import *
from vllm import LLM, SamplingParams

repe_pipeline_registry()
device = torch.device("cuda")

def comparison_gsm8k(args):
    sampling_params = SamplingParams(max_tokens=args.max_length)
    model=LLM(model=args.model_path,gpu_memory_utilization=0.90)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path,padding_side='left')
    tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    inputs=[]
    data_com=[]
    data = []

    with open('/home/chh/repos/my_ctg/results/gsm8k_act/__data1__chh__models__meta-llama__Meta-Llama-3-8B-Instruct/samples_gsm8k_cot_llama_train_2024-11-11T22-37-30.083129.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    with open('/home/chh/repos/my_ctg/instructions/template/compare_gsm8k.txt','r',encoding='utf-8') as f:
        template=f.read()
    for i in range(len(data)):
        record=data[i]['doc']
        record['result']=data[i]['exact_match']
        record['generated_text']=data[i]["resps"][0][0]
        data_com.append(record)
        if record['result']==0.0:
            inputs.append(prompt_template(tokenizer,template%(record['question'],record['answer'],record['generated_text'])))
            
    outputs = model.generate(inputs, sampling_params)
    res = [item.outputs[0].text for item in outputs]
    
    index=0
    for i in data_com:
        if i['result']==0:
            i['key']=res[index]
            index+=1
    with open(args.eval_file, 'w', encoding='utf-8') as output_file:
        json.dump(data_com, output_file, ensure_ascii=False, indent=4)

def comparison_math(args):
    sampling_params = SamplingParams(max_tokens=args.max_length)
    model=LLM(model=args.model_path,gpu_memory_utilization=0.90)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path,padding_side='left')
    tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    inputs=[]
    data_com=[]
    data = []

    with open('/home/chh/repos/my_ctg/results/math_act/__data1__chh__models__meta-llama__Meta-Llama-3-8B-Instruct/samples_minerva_math_algebra_2024-11-11T22-37-46.372276.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    with open('/home/chh/repos/my_ctg/results/math_act/__data1__chh__models__meta-llama__Meta-Llama-3-8B-Instruct/samples_minerva_math_counting_and_prob_2024-11-11T22-37-46.372276.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    with open('/home/chh/repos/my_ctg/results/math_act/__data1__chh__models__meta-llama__Meta-Llama-3-8B-Instruct/samples_minerva_math_geometry_2024-11-11T22-37-46.372276.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    with open('/home/chh/repos/my_ctg/results/math_act/__data1__chh__models__meta-llama__Meta-Llama-3-8B-Instruct/samples_minerva_math_intermediate_algebra_2024-11-11T22-37-46.372276.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    with open('/home/chh/repos/my_ctg/results/math_act/__data1__chh__models__meta-llama__Meta-Llama-3-8B-Instruct/samples_minerva_math_num_theory_2024-11-11T22-37-46.372276.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    with open('/home/chh/repos/my_ctg/results/math_act/__data1__chh__models__meta-llama__Meta-Llama-3-8B-Instruct/samples_minerva_math_prealgebra_2024-11-11T22-37-46.372276.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    with open('/home/chh/repos/my_ctg/results/math_act/__data1__chh__models__meta-llama__Meta-Llama-3-8B-Instruct/samples_minerva_math_precalc_2024-11-11T22-37-46.372276.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    with open('/home/chh/repos/my_ctg/instructions/template/compare_math.txt','r',encoding='utf-8') as f:
        template=f.read()
    for i in range(len(data)):
        record=data[i]['doc']
        record['result']=data[i]['exact_match']
        record['generated_text']=data[i]["resps"][0][0]
        data_com.append(record)
        if record['result']==0:
            inputs.append(prompt_template(tokenizer,template%(record['problem'],record['solution'],record['generated_text'])))
            
    outputs = model.generate(inputs, sampling_params)
    res = [item.outputs[0].text for item in outputs]
    
    index=0
    for i in data_com:
        if i['result']==0:
            i['key']=res[index]
            index+=1
    with open(args.eval_file, 'w', encoding='utf-8') as output_file:
        json.dump(data_com, output_file, ensure_ascii=False, indent=4)

def extract(args):
    with open('/home/chh/repos/my_ctg/results/gsm8k_act/res3.json', 'r', encoding='utf-8') as f:
        data_gsm8k = json.load(f)
    with open('./results/math_act/res2.json', 'r', encoding='utf-8') as f:
        data_math = json.load(f)
    data=data_gsm8k+data_math
    positive_samples = []
    negative_samples = []
    
    for item in data:
        if item['result']==1:
            negative_samples.append({'instruction':item['generated_text'],'output':"0" })
            continue
        content=item['key']
        # json_start = content.find('{')
        # content=item['key'][json_start:]
        pattern = r'\{[^{}]*\}'
        match = re.findall(pattern, content)
        if match:
            content=match[0]
        else:
            content+='}'
        match = re.findall(pattern, content)
        if match:
            content=match[0]
        try:
            answer = json.loads(content)
        except Exception as e:
            
            answer=None
        # print(answer)
        
        if answer:
            sentences=list(answer.values())
            positive_samples.append({'instruction':sentences[0],'output':"1" })
                
    positive_count = len(positive_samples)
    print(positive_count)
    if len(negative_samples) > positive_count:
        negative_samples = random.sample(negative_samples, positive_count*3)
    print(len(negative_samples))
    train_data=positive_samples+negative_samples
    with open('/home/chh/repos/my_ctg/sft/cls_dataset/data3.json', "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='/data1/chh/models/meta-llama/Meta-Llama-3-8B-Instruct')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_length', type=int, default=1024)
    parser.add_argument('--eval_file', type=str, default='./results/math_act/res2.json')
    parser.add_argument('--cls_dataset',type=str,default='/home/chh/repos/my_ctg/sft/cls_dataset/data.json')
    
    
    
    args = parser.parse_args()
    set_seed(args)
    
   
    # comparison_gsm8k(args)
    # comparison_math(args)
    extract(args)
    