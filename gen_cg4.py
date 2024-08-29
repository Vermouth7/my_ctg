import argparse
import json
import os

os.environ['CUDA_VISIBLE_DEVICES']='7'
import gc
import random

import numpy as np
import torch
from tqdm import tqdm
from transformers import (AutoConfig, AutoModel, AutoModelForCausalLM,
                          AutoModelForSequenceClassification, AutoTokenizer,
                          LlamaForCausalLM, LlamaTokenizer)
from utils import get_data, prompt_template, set_seed
from vllm import LLM, SamplingParams


def check_train(datum:dict, unseen_combs:list) -> bool:
    '''
    return True: datum is supposed to be in train_set;
    otherwise: datum is supposed to be hold out;
    '''
    for i in range(len(unseen_combs)):
        flag = True
        for key in unseen_combs[i].keys():
            if datum[key] != unseen_combs[i][key]:
                flag = False
        if flag == False: # at least one attribute different
            pass
        else: # in this combination, all the attributes match
            return False
    return True

def get_data_by_unseen_combs(dataset_path:str, unseen_combs:list) -> list:
    '''
    dataset_path: contains all data of this dataset
    unseen_combs: contains one or more attribute combinations 
    The functions's purpose is to filter out all data from dataset_path that contains attribute combinations present in unseen_combs.
    '''
    f = open(dataset_path, 'r')
    all_data = list()
    all_combs = list()
    for item in f.readlines():
        dic = json.loads(item)
        all_data.append(dic)
        comb = dict()
        for key in dic.keys():
            if key != 'review':
                comb[key] = dic[key]
        if comb not in all_combs:
            all_combs.append(comb)
    all_combs = list(all_combs)

    data_train = list()
    for datum in all_data:
        if check_train(datum, unseen_combs) == True:
            data_train.append(datum)
    
    return data_train, all_combs

def get_train_dataset(dataset_path:str, unseen_combs_path:str, mode:str, idx:int) -> list:
    unseen_combs_dict = {}
    f = open(unseen_combs_path, 'r')
    for item in f.readlines():
        dic = json.loads(item)
        unseen_combs = dic['unseen_combs']
        _idx = dic['idx']
        _mode = dic['mode']
        if _mode not in list(unseen_combs_dict.keys()):
            unseen_combs_dict[_mode] = list()
            unseen_combs_dict[_mode].append((unseen_combs, _mode, _idx))
        else:
            unseen_combs_dict[_mode].append((unseen_combs, _mode, _idx))
    f.close()
    assert mode in list(unseen_combs_dict.keys())
    assert idx < len(unseen_combs_dict[mode])

    unseen_combs = unseen_combs_dict[mode][idx][0]
    train_dataset, all_combs = get_data_by_unseen_combs(dataset_path=dataset_path, unseen_combs=unseen_combs)
    mode_name = unseen_combs_dict[mode][idx][1] + unseen_combs_dict[mode][idx][2]

    return train_dataset, mode_name, all_combs, unseen_combs

def main(args):
    prompts=['In summary','This essay discusses','Views on','The connection','Foundational to this is',
            'To review,','In brief,','An illustration of','Furthermore,','The central theme',
            'To conclude,','The key aspect','Prior to this','Emphasised are','To summarise',
            'The relationship','More importantly,','It has been shown','The issue focused on','In this essay',
            'Once upon a time','The book','The chicken','The city','The country',
            'The horse','The lake','The last time','The movie','The painting',
            'The pizza','The potato','The president of the country','The road','The year is 1910']
    
    model = LLM(model=args.model_path,gpu_memory_utilization=0.95)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    sampling_params = SamplingParams(temperature=0.6, top_p=0.9,max_tokens=args.max_length)
    train_dataset, _, all_combs, unseen_combs = get_train_dataset(dataset_path=args.dataset_path, unseen_combs_path=args.unseen_combs_path, mode=args.mode, idx=args.idx)
    inputs=[]
    generated_data=[]
    
    if args.mode=='Original':
        for comb in all_combs:
            matching_examples = [
            example for example in train_dataset
            if all(example.get(key) == comb[key] for key in ['sentiment', 'pronoun', 'tense'])
            ]
            for prompt in prompts:
                for _ in range(10):
                    in_context_input = "Task: write a sentence that meets the requirement of input control conditions.  \n"
                    examples = random.sample(matching_examples, 5)
                    for i,example in enumerate(examples):
                        example_comb_str = ', '.join([f"{key}: {value}" for key, value in example.items() if key != 'review'])
                        in_context_input += f"{i+1}.Input: {example_comb_str}.\nOutput: {example['review']}\n"
                    comb_str = ', '.join([f"{key}: {value}" for key, value in comb.items()])
                    in_context_input += f"6.Input: {comb_str}.\nOutput: {prompt}"
                    
                    # inputs.append(prompt_template(tokenizer,in_context_input,sys_prompt=""))
                    inputs.append(in_context_input)
                    generated_data.append(comb)
        outputs = model.generate(inputs, sampling_params)
        for output,gen in zip(outputs,generated_data):
            gen['text']=output.outputs[0].text
        with open(args.output, 'w') as outfile:
            for entry in generated_data:
                json.dump(entry, outfile)
                outfile.write('\n')
    else:
        seen_combs = [comb for comb in all_combs if comb not in unseen_combs]
    
        seen_generated_data = []
        unseen_generated_data = []

        for comb in seen_combs:
            matching_examples = [
                example for example in train_dataset
                if all(example.get(key) == comb[key] for key in ['sentiment', 'pronoun', 'tense'])
            ]
            for prompt in prompts:
                for _ in range(10):
                    in_context_input = "Task: write a sentence that meets the requirement of input control conditions. Below are some examples (Input, Output) for the task: \n"
                    examples = random.sample(matching_examples, 5)
                    for example in examples:
                        example_comb_str = ', '.join([f"{key}: {value}" for key, value in example.items() if key != 'review'])
                        in_context_input += f"Input: {example_comb_str}.\n Output: {example['review']}\n"
                    comb_str = ', '.join([f"{key}: {value}" for key, value in comb.items()])
                    in_context_input += f" Input: {comb_str}.\nOutput: {prompt}"
                    
                    inputs.append(in_context_input)
                    seen_generated_data.append(comb)
        
        outputs = model.generate(inputs, sampling_params)
        for output, gen in zip(outputs, seen_generated_data):
            gen['text'] = output.outputs[0].text
        
        inputs = []

        for comb in unseen_combs:
            for prompt in prompts:
                for _ in range(10):
                    in_context_input = "Task: write a sentence that meets the requirement of input control conditions. Below are 5 examples (Input, Output) for the task: \n"
                    examples = random.sample(train_dataset, 5)  # Randomly sample 5 examples from the entire train set
                    for example in examples:
                        example_comb_str = ', '.join([f"{key}: {value}" for key, value in example.items() if key != 'review'])
                        in_context_input += f"Input: {example_comb_str}.\n Output: {example['review']}\n"
                    comb_str = ', '.join([f"{key}: {value}" for key, value in comb.items()])
                    in_context_input += f" Input: {comb_str}.\nOutput: {prompt}"
                    
                    inputs.append(in_context_input)
                    unseen_generated_data.append(comb)
        
        outputs = model.generate(inputs, sampling_params)
        for output, gen in zip(outputs, unseen_generated_data):
            gen['text'] = output.outputs[0].text
        
        seen_output_file = args.output.replace('.json', '_seen.json')
        unseen_output_file = args.output.replace('.json', '_unseen.json')
        
        with open(seen_output_file, 'w') as outfile:
            for entry in seen_generated_data:
                json.dump(entry, outfile)
                outfile.write('\n')
        
        with open(unseen_output_file, 'w') as outfile:
            for entry in unseen_generated_data:
                json.dump(entry, outfile)
                outfile.write('\n')

    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--model_path", default='/data1/chh/models/meta-llama/Meta-Llama-3-8B-Instruct', type=str)
    parser.add_argument("--task", default='Yelp', type=str)
    parser.add_argument("--mode", default='Original', type=str, choices=['Original','Hold-Out','ACD','Few-Shot',])
    parser.add_argument("--idx", default='0', type=int)
    parser.add_argument("--output", default='', type=str)
    # decoding
    parser.add_argument("--top_p", default=0.9, type=float)
    parser.add_argument("--max_length", default=50, type=int)
    parser.add_argument("--temperature", default=0.6, type=float)
    
    parser.add_argument("--dataset_path", default="/home/chh/repos/CG4MCTG/data/Yelp/gen.jsonl", type=str)
    parser.add_argument("--unseen_combs_path", default="/home/chh/repos/CG4MCTG/data/Yelp/unseen.jsonl", type=str)
    
    args = parser.parse_args()
    args.output='./results/cg4mctg/{}-{}-{}.json'.format(args.task,args.mode,args.idx)
    set_seed(args)

    main(args)
    
