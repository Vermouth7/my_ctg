import argparse
import json
import os
import re
import time

import accelerate
import numpy as np
import sympy
import torch
import torch.distributed as dist
import torch.nn.functional as F
from accelerate import Accelerator, PartialState
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, LlamaForCausalLM, pipeline
from utils import *
from vllm import LLM, SamplingParams
from wrappedmodel import WrappedModel


def process_data(sample):
    res={}
    res['sub_ins']=[]
    res['sub_ins'].append(sample['instruction 1'])
    res['sub_ins'].append(sample['instruction 2'])
    
    return res

def get_split_hs(model,tokenizer,batch_data,batch_size):
    all_hiddens= []
    
    with open(os.path.join("/home/chh/repos/my_ctg/instructions/ifeval/ifeval_2steps_llama.json"), 'r', encoding='utf-8') as input_file:
        data=json.load(input_file)
    
    samples=[]
    for batch_group in batch_data:
        matched_samples = []
        for sample in batch_group:
            prompt = sample['prompt']  
            
            matched_sample = next((item for item in data if item['prompt'] == prompt), None)
            if matched_sample:
                matched_samples.append(matched_sample)
        samples.extend(matched_samples)
    
    for sample in accelerate.utils.tqdm(samples, desc=f"Processing dataset", unit="line"):
        res=process_data(sample)
        
        hidden_states_list = []
        for sub_instruction in res['sub_ins']:
            sub_instruction=prompt_template(tokenizer=tokenizer,message=sub_instruction)
            inputs = tokenizer(sub_instruction, return_tensors='pt')
            inputs.to(model.model.device)
            with torch.no_grad():
                outputs = model(**inputs,output_hidden_states=True)
            
            hidden_states = outputs.hidden_states
            stacked_hidden_states = torch.stack([layer_output[:, -1:, :] for layer_output in hidden_states]) # 33 1 token_pos 4096
            
            # stacked_hidden_states = torch.mean(stacked_hidden_states, dim=2, keepdim=True)
            stacked_hidden_states = torch.transpose(stacked_hidden_states, 0, 1)
            hidden_states_list.append(stacked_hidden_states)

        hidden_states_tensor = torch.stack(hidden_states_list)
        average_hidden_state = torch.mean(hidden_states_tensor, dim=0)
        average_hidden_state = average_hidden_state.squeeze(0)
        all_hiddens.append(average_hidden_state)
    all_hiddens=torch.stack(all_hiddens)
    
    batches_hidden=[all_hiddens[i:i + batch_size] for i in range(0,all_hiddens.shape[0], batch_size)]
    
    return batches_hidden
    

def vllm_gen(args):
    sampling_params = SamplingParams(temperature=0.6, top_p=0.95,max_tokens=args.max_new_tokens)
    model=LLM(model=args.model_path,gpu_memory_utilization=0.90)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path,padding_side='left')
    tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    run_results = []
    inputs=[]
    
    # data=load_dataset("/data1/chh/datasets/google/IFEval")
    # test_data=data['train']
    
    with open(os.path.join("/home/chh/repos/my_ctg/instructions/ifeval/ifeval_2steps_llama_3.json"), 'r', encoding='utf-8') as input_file:
        test_data=json.load(input_file)
    
    for i in test_data:
        instruction=prompt_template(tokenizer,i['prompt']+" "+i['instruction 1']+i['instruction 2'])
        inputs.append(instruction)
        
    
    outputs = model.generate(inputs, sampling_params)
    res = [item.outputs[0].text for item in outputs]
        
    for r,i in zip(res,test_data):
        run_results.append({'prompt':i['prompt'],'response': r})
    
    with open(args.output_file, 'w', encoding='utf-8') as file:
        for item in run_results:
            file.write(json.dumps(item) + '\n')

def CTG_hs_pal(args):
    distributed_state = PartialState()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, padding_side='left')
    model = LlamaForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16,device_map=distributed_state.device)
    tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    model = model.eval()
    model = WrappedModel(model, tokenizer)
    
    insert_layer = [18]
    layers = [i - 1 for i in insert_layer]
    print('insert layer: ', insert_layer)
    
    run_results = []
    batch_size = 1 
      
    data = load_dataset("/data1/chh/datasets/google/IFEval")
    test_data = []
    for i in data['train']:
        instruction = prompt_template(tokenizer, i['prompt'])
        test_data.append({'prompt': i['prompt'], 'temp': instruction})
    
    batches_test = [test_data[i:i + batch_size] for i in range(0, len(test_data), batch_size)]
    
    with distributed_state.split_between_processes(batches_test,apply_padding=True) as batched_prompts:
        vector_pool=get_split_hs(model,tokenizer,batched_prompts,batch_size)
        for index, item in enumerate(accelerate.utils.tqdm(batched_prompts, desc="Processing prompts")):
            inputs = [i['temp'] for i in item]
            vector = vector_pool[index]
            
            model.reset()
            model.set_controller(layer_ids=layers, activations=vector)
            model.set_pos(inputs)
            
            inputs = tokenizer(inputs, return_tensors="pt", padding=True).to(distributed_state.device)
            input_ids_cutoff = inputs.input_ids.size(dim=1)
            
            generated_ids = model.generate(
                **inputs,
                use_cache=False,
                max_new_tokens=args.max_new_tokens,
                temperature=0.6,
                top_p=0.90,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
                mode=2,
                output_hidden_states=True
            )
            
            res = tokenizer.batch_decode(
                [ids[input_ids_cutoff:] for ids in generated_ids],
                skip_special_tokens=True,
            )
            
            for r, i in zip(res, item):
                run_results.append({'prompt': i['prompt'], 'response': r})
    torch.distributed.barrier()  
    
    res_gather = accelerate.utils.gather_object(run_results)
    torch.distributed.barrier()  
    if distributed_state.is_main_process:
        memo = set()
        final_res = []
        for result in res_gather:
            if result['prompt'] not in memo:
                final_res.append(result)
                memo.add(result['prompt'])
        with open(args.output_file, 'w', encoding='utf-8') as file:
            for item in final_res:
                file.write(json.dumps(item) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='/data1/chh/models/meta-llama/Meta-Llama-3-8B-Instruct')
    parser.add_argument('--output_file', type=str, default='./results/ifeval/res7.jsonl')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_new_tokens', type=int, default=1024)
        
    args = parser.parse_args()
    accelerate.utils.set_seed(args.seed)
    
    # for baseline
    vllm_gen(args)

    # CTG_hs_pal(args)

# python -m instruction_following_eval.evaluation_main \
#   --input_data=/home/chh/repos/my_ctg/instruction_following_eval/data/input_data.jsonl \
#   --input_response_data=./results/ifeval/res7.jsonl \
#   --output_dir=./results/ifeval/