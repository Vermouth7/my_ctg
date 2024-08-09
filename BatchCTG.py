import os

os.environ['CUDA_VISIBLE_DEVICES']='0'
import argparse
import json
import time

import numpy as np
import torch
from batch_repe import repe_pipeline_registry
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, LlamaForCausalLM, pipeline
from utils import (compute_metric, get_test_data, load_eval_models,
                   process_test_datset, set_seed)

repe_pipeline_registry()
device = torch.device("cuda")



"""
need get_hidden_states
need compute baseline
need handle dataset and different token positions
"""

def get_condition_output(model,tokenizer,prompts,num_condition,pos):
    all_hiddens, all_logits = [], []
    batches = [prompts[i:i + num_condition] for i in range(0, len(prompts), num_condition)]
    
    for batch_input in tqdm(batches):
        encoded_inputs = tokenizer(batch_input, return_tensors="pt",padding=True)
        encoded_inputs.to(device)
        
        if not all_hiddens:
            print(tokenizer.convert_ids_to_tokens(encoded_inputs["input_ids"][0]))

        with torch.no_grad():
            outputs = model(**encoded_inputs, output_hidden_states=True, return_dict=True)
            logits = outputs.logits[:, pos, :]
            hidden_states = outputs.hidden_states
            
        stacked_hidden_states = torch.stack([layer_output[:, pos:, :] for layer_output in hidden_states])
        # stacked_hidden_states = torch.transpose(stacked_hidden_states, 0, 1)

        mean_hidden_states = stacked_hidden_states.mean(dim=1, keepdim=True)
        mean_logits = logits.mean(dim=0, keepdim=True)

        all_hiddens.append(mean_hidden_states)
        all_logits.append(mean_logits)

    hiddens = torch.cat(all_hiddens, dim=1)
    logits = torch.cat(all_logits, dim=0)

    return hiddens, logits

def CTG_hs(model,tokenizer,task,output):
    run_results = {}

    start_time = time.time()
    acc = []
        
    # compute_baseline(model,tokenizer,task_name,test_df,eval_models)
    # best_layer = get_layer(tokenizer=tokenizer,model=model,task_name=task_name,eval_models=eval_models)
    
    insert_layer=12
    print('insert layer: ',insert_layer)
    run_results[insert_layer]=[]
    test_data,split,num_condition=process_test_datset(tokenizer,args.task)
    hiddens,logits = get_condition_output(model,tokenizer,split,num_condition,pos=-1)
    # print(hiddens.shape)
    # print(logits.shape)
    
    control_pipeline = pipeline(
        "ctg-control", 
        model=model, 
        tokenizer=tokenizer, 
        layers=insert_layer)
    
    
    
    print('Now performing aggregation on layer {}'.format(insert_layer))
    
    for item,hidden in tqdm(zip(test_data,hiddens), desc="Processing prompts"):
        
        control_outputs = control_pipeline(item['prompt'], activations=hidden, batch_size=1, max_new_tokens=512, do_sample=True)
        control_outputs = [item[0]['generated_text'] for item in control_outputs]
        for result in control_outputs:
            run_results[insert_layer].append({'text': result, 'label1': item['label1'],'label2':item['label2']})
    
    with open(output, 'w') as f:
        json.dump(run_results, f, ensure_ascii=False, indent=2)

    # score, layer = compute_metric(output,task,test_data)
    # acc.append(score)

    end_time = time.time()
    print("total run time %.2f" % (end_time - start_time))
    
    # acc = np.array(acc)
    # acc_mean = np.mean(acc)
    
    # print("Multi CTG Acc:"+str(acc_mean))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='/data1/chh/models/meta-llama/Meta-Llama-3-8B-Instruct')
    # parser.add_argument('--ckpt_dir', type=str, default='/data1/chh/models/model_sft/llama3_lora_sft')
    parser.add_argument('--task', type=str, default='multi')
    parser.add_argument('--output', type=str, default='./batch_ctg/test.json')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    set_seed(args)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path,padding_side='left')
    model = LlamaForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16).to(device)

    tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    model = model.eval()
    CTG_hs(model,tokenizer,args.task,args.output)
    