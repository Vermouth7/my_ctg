import argparse
import json
import os
import re
import time

import accelerate
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator, PartialState
from batch_repe import repe_pipeline_registry
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, LlamaForCausalLM, pipeline
from utils import *
from vllm import LLM, SamplingParams
from wrappedmodel import WrappedModel

constraint_types=['content', 'situation', 'style', 'format', 'example', 'mixed']

def extract_and_store(text):
    # pattern = r'\d+\.\s*\[(.*?)\](?=\s*\d+\.|\s*$)'
    pattern = r'\[\s*(.*?)\s*\]'
    matches = re.findall(pattern, text)
    
    matches = [match.strip() for match in matches]

    result = {
        "original_text": text,
        "extracted_contents": matches
    }
    
    return result

def process_data(sample):
    res={}
    res['sub_ins']=[]
    res['sub_ins'].append(sample['instruction 1'])
    res['sub_ins'].append(sample['instruction 2'])
    
    return res

def get_split_hs(model,tokenizer,constraint,batch_data,batch_size):
    all_hiddens= []
    data=[]
    with open(os.path.join("/home/chh/repos/my_ctg/instructions/followbench5/{}_constraint_split.jsonl".format(constraint)), 'r', encoding='utf-8') as input_file:
        for line in tqdm(input_file, desc=f"Processing {constraint}", unit="line"):
            temp=json.loads(line.strip())
            data.append(temp)
    samples=[]
    for batch_group in batch_data:
        matched_samples = []
        for sample in batch_group:
            question = sample['prompt_new']  
            
            matched_sample = next((item for item in data if item['prompt_new'] == question), None)
            if matched_sample:
                matched_samples.append(matched_sample)
        samples.extend(matched_samples)
        
    for sample in accelerate.utils.tqdm(samples, desc=f"Processing dataset", unit="line"):
        
        # res=extract_and_store(temp['split_ins'])
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

def get_split_hs_nomean(model,tokenizer,constraint,pos):
    all_hiddens= {}
    
    with open(os.path.join("/home/chh/repos/my_ctg/instructions/followbench2/{}_constraint_split.jsonl".format(constraint)), 'r', encoding='utf-8') as input_file:
        for idx,line in enumerate(tqdm(input_file, desc=f"Processing {constraint}", unit="line")):
            temp=json.loads(line.strip())
            res=extract_and_store(temp['split_ins'])
            
            hidden_states_list = []
            for sub_instruction in res['extracted_contents']:
                sub_instruction=prompt_template(tokenizer=tokenizer,message=sub_instruction)
                inputs = tokenizer(sub_instruction, return_tensors='pt')
                inputs.to(device)
                with torch.no_grad():
                    outputs = model(**inputs,output_hidden_states=True)
                
                hidden_states = outputs.hidden_states
                stacked_hidden_states = torch.stack([layer_output[:, -1:, :] for layer_output in hidden_states]) # 33 1 token_pos 4096
                
                # stacked_hidden_states = torch.mean(stacked_hidden_states, dim=2, keepdim=True)
                stacked_hidden_states = torch.transpose(stacked_hidden_states, 0, 1) # 1 33 1 4096
                hidden_states_list.append(stacked_hidden_states)
    
            hidden_states_tensor = torch.stack(hidden_states_list) # num_condi 1 33 1 4096
            hidden_states_tensor = hidden_states_tensor.squeeze(1)
            all_hiddens[idx]=hidden_states_tensor
    # for k,v in all_hiddens.items():
    #     print(k,v.shape)
    return all_hiddens

def CTG_hs(args):
    distributed_state = PartialState()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path,padding_side='left')
    model = LlamaForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16,device_map=distributed_state.device)
    
    tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    model = model.eval()
    model = WrappedModel(model, tokenizer)
    
    insert_layer=[18]
    layers = [i - 1 for i in insert_layer]

    print('insert layer: ',insert_layer)
    
    for constraint in constraint_types:
        run_results = []
        batch_size = 1
        test_data=[]

        with open(os.path.join("/home/chh/repos/my_ctg/instructions/followbench2/{}_constraint.jsonl".format(constraint)), 'r', encoding='utf-8') as test_file:
            for line in test_file:
                temp_dict = json.loads(line.strip())
                prompt=temp_dict['prompt_new']
                prompt_tem=prompt_template(tokenizer,prompt)
                test_data.append({'prompt_new':prompt,'prompt_input':prompt_tem})
        
        batches_test = [test_data[i:i + batch_size] for i in range(0, len(test_data), batch_size)]
        
        with distributed_state.split_between_processes(batches_test,apply_padding=True) as batched_prompts:
            vector_pool=get_split_hs(model,tokenizer,constraint,batched_prompts,batch_size)
            
            for index, item in enumerate(accelerate.utils.tqdm(batched_prompts, desc="Processing prompts")):
                inputs=[i['prompt_input'] for i in item]
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
                    mode=0,
                    output_hidden_states=True
                )
                
                res = tokenizer.batch_decode(
                    [ids[input_ids_cutoff:] for ids in generated_ids],
                    skip_special_tokens=True,
                )
                
                for r,i in zip(res,item):
                    run_results.append({'prompt_new':i['prompt_new'],'result': r})
                distributed_state.wait_for_everyone()
                
        distributed_state.wait_for_everyone()
        res_gather = accelerate.utils.gather_object(run_results)
        
        if distributed_state.is_main_process:
            if not os.path.exists(args.output_folder):
                os.makedirs(args.output_folder)
            memo = set()
            final_res = []
            for result in res_gather:
                if result['prompt_new'] not in memo:
                    final_res.append(result)
                    memo.add(result['prompt_new'])
            with open(os.path.join(args.output_folder, f"{os.path.basename(args.model_path)}_{constraint}_constraint.jsonl"), 'w', encoding='utf-8') as output_file:
                for d in final_res:
                    output_file.write(json.dumps(d) + "\n")
        distributed_state.wait_for_everyone()

def vllm_gen(args):
    sampling_params = SamplingParams(temperature=0.6, top_p=0.9,max_tokens=args.max_new_tokens)
    model=LLM(model=args.model_path,gpu_memory_utilization=0.90)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path,padding_side='left')
    tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    
    
    for constraint in constraint_types:
        run_results = []
        test_data=[]

        with open(os.path.join("/home/chh/repos/my_ctg/instructions/followbench2/{}_constraint.jsonl".format(constraint)), 'r', encoding='utf-8') as test_file:
            for line in test_file:
                temp_dict = json.loads(line.strip())
                prompt=temp_dict['prompt_new']
                prompt_tem=prompt_template(tokenizer,prompt)
                test_data.append({'prompt_new':prompt,'prompt_input':prompt_tem})
        inputs=[i['prompt_input'] for i in test_data]
        outputs = model.generate(inputs, sampling_params)
        res = [item.outputs[0].text for item in outputs]
        for r,i in zip(res,test_data):
            run_results.append({'prompt_new':i['prompt_new'],'result': r})
            
        with open(os.path.join(args.output_folder, f"{os.path.basename(args.model_path)}_{constraint}_constraint.jsonl"), 'w', encoding='utf-8') as output_file:
                for d in run_results:
                    output_file.write(json.dumps(d) + "\n")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='/data1/chh/models/meta-llama/Meta-Llama-3-8B-Instruct')
    parser.add_argument('--output_folder', type=str, default='./results/followbench/res17')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_new_tokens', type=int, default=1024)
    
    args = parser.parse_args()
    
    set_seed(args)
    CTG_hs(args)
    # vllm_gen(args)