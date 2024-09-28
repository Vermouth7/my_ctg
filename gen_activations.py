import argparse
import json
import os

os.environ['CUDA_VISIBLE_DEVICES']='0'

import re
import time
from difflib import SequenceMatcher

import numpy as np
import torch
import torch.nn.functional as F
from batch_repe import repe_pipeline_registry
from datasets import load_dataset
from tqdm import tqdm
from transformers import (AutoModel, AutoTokenizer, LlamaForCausalLM,
                          StoppingCriteria, StoppingCriteriaList, pipeline)
from utils import *
from vllm import LLM, SamplingParams

repe_pipeline_registry()
device = torch.device("cuda")
    
def gen(args):
    model = LlamaForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16).to(device)
    model = model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path,padding_side='left')
    tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    run_results = []
    batch_size = 8
    test_data=[]
    prompt="You're a mathematician who's good at reasoning. Answer the following questions using detailed reasoning steps, and you MUST write the answer as an integer after '####'.\nQuestion: {question}"
    
    train_data=load_dataset("/data1/chh/datasets/openai/gsm8k",'main')
    train_data=train_data['train'].to_pandas().to_dict(orient='records')[:1000]
    for i in train_data:
        i['new_q']=prompt_template(tokenizer,prompt.format(question=i['question']))
        
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
        
        for j, generated_output in enumerate(outputs.sequences):
            input_length = input_ids[j].size(0)
            
            generated_tokens = generated_output[input_length:]
            decoded_output = tokenizer.decode(generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            
            train_data[i + j]['generated_text'] = decoded_output
            all_hidden_states[i+j]=sample_hs[j]
            del train_data[i + j]['new_q']
    
    torch.save(all_hidden_states, args.classifier_data)
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
    model=LLM(model=args.model_path,gpu_memory_utilization=0.90)
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
            inputs.append(prompt_template(tokenizer,template.format(question=record['question'],answer=record['answer'],generated=record['generated_text'])))
    
    outputs = model.generate(inputs, sampling_params)
    res = [item.outputs[0].text for item in outputs]
    
    for r,i in zip(res,data_com):
        i['key']=r
    
    with open(args.eval_file, 'w', encoding='utf-8') as output_file:
        json.dump(data_com, output_file, ensure_ascii=False, indent=4)

def find_best_match(sentence_tokens, text_tokens):
    best_match_start = 0
    best_match_length = 0
    best_match_ratio = 0
    
    for i in range(len(text_tokens) - len(sentence_tokens) + 1):
        window = text_tokens[i:i + len(sentence_tokens)]
        match_ratio = SequenceMatcher(None, sentence_tokens, window).ratio()
        
        if match_ratio > best_match_ratio:
            best_match_ratio = match_ratio
            best_match_start = i
            best_match_length = len(sentence_tokens)
    
    return best_match_start, best_match_start + best_match_length - 1, best_match_ratio

def find_token_pos(text, sentences, tokenizer):
    tokenized_text = tokenizer.encode(text, add_special_tokens=False)  
    positions = []
    
    for sentence in sentences:
        tokenized_sentence = tokenizer.encode(sentence, add_special_tokens=False)  
        
        start, end, match_ratio = find_best_match(tokenized_sentence, tokenized_text)
        positions.append((sentence, start, end, match_ratio))
    
    return positions

def extract_hs(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path,padding_side='left')
    tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    train_data=torch.load(args.classifier_data)
    with open(args.eval_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for item in data:
        if item['result']==1:
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
            ref=item['generated_text']
            token_pos = find_token_pos(ref, sentences, tokenizer)
            
            for sentence, start, end, match_ratio in token_pos:
                sample_data=train_data[item['id']][start:end]
                ## next: stack tensor and label it as 1 
                ## sample some 0 label
                ## tran the classifier
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='/data1/chh/models/meta-llama/Meta-Llama-3-8B-Instruct')

    # parser.add_argument('--model_path', type=str, default='/data1/chh/models/model_sft/llama3-8b/merge/qwen/sft3')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--eval_file',type=str,default='./results/gsm8k_act/res1.json')
    parser.add_argument('--output_folder', type=str, default='./results/gsm8k_act/res1.json')
    parser.add_argument('--classifier_data',type=str,default='/home/chh/repos/my_ctg/sft/classifier/demo.pt')
    args = parser.parse_args()
    set_seed(args)
    
    # gen(args)
    # eval_func(args)
    comparison(args)
    # extract_hs(args)
    