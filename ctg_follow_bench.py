import os

os.environ['CUDA_VISIBLE_DEVICES']='2'
import argparse
import json
import re
import time

import numpy as np
import torch
import torch.nn.functional as F
from batch_repe import repe_pipeline_registry
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, LlamaForCausalLM, pipeline
from utils import *

repe_pipeline_registry()
device = torch.device("cuda")
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

def get_split_hs(model,tokenizer,constraint,pos):
    all_hiddens= []
    
    with open(os.path.join("/home/chh/repos/my_ctg/instructions/followbench2/{}_constraint_split.jsonl".format(constraint)), 'r', encoding='utf-8') as input_file:
        for line in tqdm(input_file, desc=f"Processing {constraint}", unit="line"):
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
                stacked_hidden_states = torch.transpose(stacked_hidden_states, 0, 1)
                hidden_states_list.append(stacked_hidden_states)
    
            hidden_states_tensor = torch.stack(hidden_states_list)
            average_hidden_state = torch.mean(hidden_states_tensor, dim=0)
            average_hidden_state = average_hidden_state.squeeze(0)
            all_hiddens.append(average_hidden_state)
    all_hiddens=torch.stack(all_hiddens)
    return all_hiddens

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

def CTG_hs(model,tokenizer,args):
    start_time = time.time()
        
    # best_layer = get_insert_layer(model,tokenizer,task,output)
    
    insert_layer=list(range(16, 21))

    print('insert layer: ',insert_layer)
    
    # print(logits.shape)
    # print(hiddens.shape)
    
    control_pipeline = pipeline(
        "ctg-control", 
        model=model, 
        tokenizer=tokenizer, 
        layers=insert_layer,
        device=device)
    
    
    for constraint in constraint_types:
        run_results = []
        batch_size = 1
        test_data=[]
        split_hiddens= get_split_hs(model,tokenizer,constraint,-1)

        with open(os.path.join("/home/chh/repos/my_ctg/instructions/followbench2/{}_constraint.jsonl".format(constraint)), 'r', encoding='utf-8') as test_file:
            for line in test_file:
                temp_dict = json.loads(line.strip())
                prompt=temp_dict['prompt_new']
                prompt_tem=prompt_template(tokenizer,prompt)
                test_data.append({'prompt_new':prompt,'prompt_input':prompt_tem})
        
        batches_test = [test_data[i:i + batch_size] for i in range(0, len(test_data), batch_size)]
        batches_hidden=[split_hiddens[i:i + batch_size] for i in range(0,split_hiddens.shape[0], batch_size)]

        for index,item in enumerate(tqdm(batches_test, desc="Processing prompts")):
            inputs=[i['prompt_input'] for i in item]
            vector=batches_hidden[index]
            # res = control_pipeline(inputs, activations=vector,token_pos=list(range(-vector.size(dim=0), 0)),batch_size=batch_size, max_new_tokens=args.max_length)
            # res = control_pipeline(inputs, activations=None,token_pos=-1,batch_size=batch_size, max_new_tokens=args.max_length)
            res = control_pipeline(inputs, activations=vector,token_pos=-1,batch_size=batch_size, max_new_tokens=args.max_length)
            
            res = [item[0]['generated_text'] for item in res]
            for r,i in zip(res,item):
                run_results.append({'prompt_new':i['prompt_new'],'result': r})
        
        with open(os.path.join(args.output_folder, f"{os.path.basename(args.model_path)}_{constraint}_constraint.jsonl"), 'w', encoding='utf-8') as output_file:
            for d in run_results:
                output_file.write(json.dumps(d) + "\n")

    

    end_time = time.time()
    print("total run time %.2f" % (end_time - start_time))
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='/data1/chh/models/meta-llama/Meta-Llama-3-8B-Instruct')

    # parser.add_argument('--model_path', type=str, default='/data1/chh/models/model_sft/llama3-8b/merge/qwen/sft3')
    parser.add_argument('--output_folder', type=str, default='./results/followbench/res12')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_length', type=int, default=1024)
    
    args = parser.parse_args()
    set_seed(args)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path,padding_side='left')
    model = LlamaForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16).to(device)

    tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    model = model.eval()
    CTG_hs(model,tokenizer,args)
    # for constraint in constraint_types:
    #     split_hiddens= get_split_hs_nomean(model,tokenizer,constraint,-1)

    