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

def nshot_chats(tokenizer,nshot_data: list, n: int, question: str) -> dict:

    def question_prompt(s):
        return f'Question: {s}'

    def answer_prompt(s):
        return f'Answer: {s}'

    chats = []
    random.seed(42)
    for qna in nshot_data[:n]:
        chats.append(
            {"role": "user", "content": question_prompt(qna["question"])})
        chats.append(
            {"role": "assistant", "content": answer_prompt(qna["answer"])})

    chats.append({"role": "user", "content": "Let's think step by step. At the end, you MUST write the answer as an integer after '####'\n."+question_prompt(question)})

    return tokenizer.apply_chat_template(
        chats,
        tokenize=False,
        add_generation_prompt=True,
    )


def gen(args):
    model = LlamaForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16).to(device)
    model = model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path,padding_side='left')
    tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    run_results = []
    batch_size = 2
    test_data=[]
    prompt="You're a mathematician who's good at reasoning. Answer the following questions using detailed reasoning steps, and you MUST write the answer as an integer after '####'.\nQuestion: {question}"
    
    train_data=load_dataset("/data1/chh/datasets/openai/gsm8k",'main')
    train_data=train_data['train'].to_pandas().to_dict(orient='records')
    for i in train_data:
        # i['new_q']=prompt_template(tokenizer,prompt.format(question=i['question']))
        i['new_q']=nshot_chats(tokenizer,nshot_data=train_data, n=8, question=i['question'])
        
    inputs=[i['new_q'] for i in train_data]
    
    encodings = tokenizer(inputs, padding=True, truncation=True, return_tensors='pt').to(device)
    all_hidden_states = {}
    for i in tqdm(range(0, len(inputs), batch_size)):
        input_ids = encodings['input_ids'][i:i+batch_size]
        attention_mask = encodings['attention_mask'][i:i+batch_size]
        outputs = model.generate(input_ids, attention_mask=attention_mask,
                                 max_new_tokens=args.max_length,
                                 return_dict_in_generate=True,
                                 output_hidden_states=True)
        
        sample_hs=[]
        for token in range(len(outputs.hidden_states)):
            sample_hs.append(outputs.hidden_states[token][-1][:,-1])
        sample_hs=torch.stack(sample_hs,dim=0)
        sample_hs=sample_hs.transpose(0,1)
        sample_hs = sample_hs.cpu()
        for j, generated_output in enumerate(outputs.sequences):
            input_length = input_ids[j].size(0)
            
            generated_tokens = generated_output[input_length:]
            decoded_output = tokenizer.decode(generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            
            train_data[i + j]['generated_text'] = decoded_output
            all_hidden_states[i+j]=sample_hs[j]
            del train_data[i + j]['new_q']
    
    torch.save(all_hidden_states, args.original_data)
    with open(args.output_folder, 'w', encoding='utf-8') as output_file:
        json.dump(train_data, output_file, ensure_ascii=False, indent=4)


def extract_ans_from_response(answer: str, eos=None):
    if eos:
        answer = answer.split(eos)[0].strip()

    answer = answer.split('####')[-1].strip()

    for remove_char in [',', '$', '%', 'g']:
        answer = answer.replace(remove_char, '')

    try:
        return int(answer)
    except ValueError:
        return answer

def eval_func(args):
    with open(args.eval_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    correct_count = 0
    total_count = len(data)

    for i, record in enumerate(data):
        answer = record.get('answer', '')
        reference = record.get('generated_text', '')

        answer_number = extract_ans_from_response(answer)
        reference_number = extract_ans_from_response(reference)

        if answer_number == reference_number:
            correct_count += 1
            record['result']=1
        else:
            record['result']=0

    accuracy = correct_count / total_count * 100
    print(f"Acc: {accuracy:.2f}%")
    with open(args.eval_file, 'w', encoding='utf-8') as output_file:
        json.dump(data, output_file, ensure_ascii=False, indent=4)
def comparison(args):
    sampling_params = SamplingParams(temperature=0.6, top_p=0.9,max_tokens=args.max_length)
    model=LLM(model=args.model_path,gpu_memory_utilization=0.80)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path,padding_side='left')
    tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    inputs=[]
    data_com=[]
    with open(args.eval_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    with open('/home/chh/repos/my_ctg/instructions/template/compare_gsm8k.txt','r',encoding='utf-8') as f:
        template=f.read()
    for i in range(len(data)):
        record=data[i]
        record['id']=i
        data_com.append(record)
        if record['result']==0:
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

def extract_hs(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path,padding_side='left')
    tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    
    with open(args.eval_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    positive_samples = []
    negative_samples = []
    
    for item in data:
        if item['result']==1:
            negative_samples.append({'instruction':item['generated_text'],'output':0 })
            continue
        content=item['key']
        # json_start = content.find('{')
        # content=item['key'][json_start:]
        pattern = r'\{[^{}]*\}'
        match = re.findall(pattern, content)
        if match:
            content=match[0]
        try:
            if content.startswith("```json"): 
                content = content[7:-3].strip()
                answer = json.loads(content)
            else:
                answer = json.loads(content)
        except Exception as e:
                # print(f"json failed to parse: {e}")
                # print(f"content: {content}")
                answer=None
        # print(answer)
        
        if answer:
            sentences=list(answer.values())
            for sentence in sentences:
                positive_samples.append({'instruction':sentence,'output':1 })
                
    positive_count = len(positive_samples)
    print(positive_count)
    print(len(negative_samples))
    # if len(negative_samples) > positive_count:
    #     negative_samples = random.sample(negative_samples, positive_count)
    train_data=positive_samples+negative_samples
    with open(args.cls_dataset, "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=4)
        
def extract_hs_math(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path,padding_side='left')
    tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    
    with open('./results/math_act/res1.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    positive_samples = []
    negative_samples = []
    
    for item in data:
        if item['result']==1:
            negative_samples.append({'instruction':item['generated_text'],'output':0 })
            continue
        content=item['key']
        json_start = content.find('{')
        content=item['key'][json_start:]
        try:
            if content.startswith("```json"): 
                content = content[7:-3].strip()
                answer = json.loads(content)
            else:
                answer = json.loads(content)
        except Exception as e:
                # print(f"json failed to parse: {e}")
                # print(f"content: {content}")
                answer=None
                
        if answer:
            sentences=list(answer.values())
            for sentence in sentences:
                positive_samples.append({'instruction':sentence,'output':1 })
                
    positive_count = len(positive_samples)
    print(positive_count)
    print(len(negative_samples))
    if len(negative_samples) > positive_count:
        negative_samples = random.sample(negative_samples, positive_count)
    train_data=positive_samples+negative_samples
    with open('/home/chh/repos/my_ctg/sft/cls_dataset/data2.json', "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='/data1/chh/models/meta-llama/Meta-Llama-3-8B-Instruct')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--eval_file',type=str,default='./results/gsm8k_act/res3.json')
    parser.add_argument('--output_folder', type=str, default='./results/gsm8k_act/res3.json')
    parser.add_argument('--cls_dataset',type=str,default='/home/chh/repos/my_ctg/sft/cls_dataset/data2.json')
    
    
    
    args = parser.parse_args()
    set_seed(args)
    
    # gen(args)
    # eval_func(args)
    # comparison(args)
    # extract_hs(args)
    # train_classifier_LR(args)
    # train_classifier_mlp(args)