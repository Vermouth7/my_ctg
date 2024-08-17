import os

os.environ['CUDA_VISIBLE_DEVICES']='2'
import argparse
import json
import time

import numpy as np
import torch
from batch_repe import repe_pipeline_registry
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, LlamaForCausalLM, pipeline
from utils import compute_metric, process_test_datset, set_seed

repe_pipeline_registry()
device = torch.device("cuda")


def get_condition_output(model,tokenizer,prompts,num_condition,pos):
    all_hiddens, all_logits, split_hiddens= [], [], []
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
        stacked_hidden_states = torch.transpose(stacked_hidden_states, 0, 1)
        
        mean_hidden_states = stacked_hidden_states.mean(dim=0, keepdim=True)
        mean_logits = logits.mean(dim=0, keepdim=True)
        

        all_hiddens.append(mean_hidden_states)
        all_logits.append(mean_logits)
        split_hiddens.append(stacked_hidden_states)
    hiddens = torch.cat(all_hiddens, dim=0)
    logits = torch.cat(all_logits, dim=0)
    split_hiddens = torch.cat(split_hiddens, dim=0)

    return hiddens, logits,split_hiddens

def get_insert_layer(model,tokenizer,task,output):
    val_path='/home/chh/repos/my_ctg/instructions/val/multi_val.jsonl'
    output_file=os.path.join(os.path.dirname(output), 'layer_'+os.path.basename(output))
    print(output_file)
    run_results = {}    
    test_data,split,num_condition=process_test_datset(tokenizer,args.task,val_path)
    
    hiddens,logits,_ = get_condition_output(model,tokenizer,split,num_condition,pos=-1)
    # print(logits.shape)
    # print(hiddens.shape)
    
    for insert_layer in range(1,model.config.num_hidden_layers+1):
        run_results[insert_layer]=[]
        control_pipeline = pipeline(
            "ctg-control", 
            model=model, 
            tokenizer=tokenizer, 
            layers=insert_layer,
            device=device)
    
    
        batch_size = 16
        batches_test = [test_data[i:i + batch_size] for i in range(0, len(test_data), batch_size)]
        batches_hidden=[hiddens[i:i + batch_size] for i in range(0,hiddens.shape[0], batch_size)]

        
        print('Now performing aggregation on layer {}'.format(insert_layer))
        for index,item in enumerate(tqdm(batches_test, desc="Processing prompts")):
            
            inputs=[i['prompt'] for i in item]
            res = control_pipeline(inputs, activations=batches_hidden[index],token_pos=-1,batch_size=batch_size, max_new_tokens=128)
            res = [item[0]['generated_text'] for item in res]
            for r,i in zip(res,item):
                run_results[insert_layer].append({'text': r, 'label1': i['label1'],'label2':i['label2']})
        
    with open(output_file, 'w') as f:
        json.dump(run_results, f, ensure_ascii=False, indent=2)

    score, layer = compute_metric(output_file,task)
    
    return layer

def CTG_hs_split(model,tokenizer,task,output):
    run_results = {}

    start_time = time.time()
    acc = []
        
    # compute_baseline(model,tokenizer,task_name,test_df,eval_models)
    # best_layer = get_layer(tokenizer=tokenizer,model=model,task_name=task_name,eval_models=eval_models)
    
    insert_layer=12
    print('insert layer: ',insert_layer)
    run_results[insert_layer]=[]
    test_data,split,num_condition=process_test_datset(tokenizer,args.task,'/home/chh/repos/my_ctg/instructions/test/multi_lite.jsonl')
   
    hiddens,logits,split_hs = get_condition_output(model,tokenizer,split,num_condition,pos=-1)
    # print(logits.shape)
    # print(hiddens.shape)
    
    
    control_pipeline = pipeline(
        "ctg-control", 
        model=model, 
        tokenizer=tokenizer, 
        layers=insert_layer,
        device=device)
    
    
    batch_size = 8
    batches_test = [test_data[i:i + batch_size] for i in range(0, len(test_data), batch_size)]
    batches_split=[split_hs[i:i + batch_size*2] for i in range(0,split_hs.shape[0], batch_size*2)]

    
    print('Now performing aggregation on layer {}'.format(insert_layer))
    for index,item in enumerate(tqdm(batches_test, desc="Processing prompts")):
        inputs=[i['prompt'] for i in item]
        res = control_pipeline(inputs, activations=batches_split[index],token_pos=[-2,-3],batch_size=batch_size, max_new_tokens=128)
        res = [item[0]['generated_text'] for item in res]
        for r,i in zip(res,item):
            run_results[insert_layer].append({'text': r, 'label1': i['label1'],'label2':i['label2']})
    
    with open(output, 'w') as f:
        json.dump(run_results, f, ensure_ascii=False, indent=2)

    score, layer = compute_metric(output,task)
    acc.append(score)

    end_time = time.time()
    print("total run time %.2f" % (end_time - start_time))
    
    acc = np.array(acc)
    acc_mean = np.mean(acc)
    
    print("Multi CTG Acc:"+str(acc_mean))
    
    
def CTG_hs(model,tokenizer,task,output):
    run_results = {}

    start_time = time.time()
    acc = []
        
    best_layer = get_insert_layer(model,tokenizer,task,output)
    
    insert_layer=best_layer
    print('insert layer: ',insert_layer)
    run_results[insert_layer]=[]
    test_data,split,num_condition=process_test_datset(tokenizer,args.task,'/home/chh/repos/my_ctg/instructions/test/multi_lite.jsonl')
    test_data=test_data
    split=split
    hiddens,logits,_ = get_condition_output(model,tokenizer,split,num_condition,pos=-1)
    # print(logits.shape)
    # print(hiddens.shape)
    
    
    control_pipeline = pipeline(
        "ctg-control", 
        model=model, 
        tokenizer=tokenizer, 
        layers=insert_layer,
        device=device)
    
    
    batch_size = 16
    batches_test = [test_data[i:i + batch_size] for i in range(0, len(test_data), batch_size)]
    batches_hidden=[hiddens[i:i + batch_size] for i in range(0,hiddens.shape[0], batch_size)]

    
    print('Now performing aggregation on layer {}'.format(insert_layer))
    for index,item in enumerate(tqdm(batches_test, desc="Processing prompts")):
        
        inputs=[i['prompt'] for i in item]
        res = control_pipeline(inputs, activations=batches_hidden[index],token_pos=-1,batch_size=batch_size, max_new_tokens=128)
        res = [item[0]['generated_text'] for item in res]
        for r,i in zip(res,item):
            run_results[insert_layer].append({'text': r, 'label1': i['label1'],'label2':i['label2']})
    
    with open(output, 'w') as f:
        json.dump(run_results, f, ensure_ascii=False, indent=2)

    score, layer = compute_metric(output,task)
    acc.append(score)

    end_time = time.time()
    print("total run time %.2f" % (end_time - start_time))
    
    acc = np.array(acc)
    acc_mean = np.mean(acc)
    
    print("Multi CTG Acc:"+str(acc_mean))
    
    
def CTG_logits(model,tokenizer,task,output,max_length):
    run_results = {}
    acc = []
    group_data=[]
    run_results[0]=[]
    test_data,split,num_condition=process_test_datset(tokenizer,args.task,'/home/chh/repos/my_ctg/instructions/test/multi_lite.jsonl')
    # test_data=test_data[:100]
    # split=split[:200]
    batch_size = 1
    batches_test = [test_data[i:i + batch_size] for i in range(0, len(test_data), batch_size)]
    batch_split=[(split[i],split[i+1]) for i in range(0,len(split),2)]
    
    for i in range(0,num_condition):
        temp=[item[i] for item in batch_split]
        batch_temp=[temp[i:i+batch_size]for i in range(0,len(temp),batch_size)]
        group_data.append(batch_temp)
    
    
    for index,item in enumerate(tqdm(batches_test, desc="Processing prompts")):
        
        inputs_test=[i['prompt'] for i in item]
        encoded_test=tokenizer(inputs_test,return_tensors="pt",padding=True)
        input_ids_test = encoded_test["input_ids"].to(device)
        attention_mask_test = encoded_test["attention_mask"].to(device)
        
        input_ids_group=[]
        attention_mask_group=[]
        for i in range(0,num_condition):
            tmp=tokenizer(group_data[i][index],return_tensors='pt',padding=True)
            input_ids_group.append(tmp["input_ids"].to(device))
            attention_mask_group.append((tmp["input_ids"]).to(device))
            
        pre_token = input_ids_test.shape[1]
        
        for _ in range(max_length):
            with torch.no_grad():
                logits_test = model(input_ids=input_ids_test,attention_mask=attention_mask_test, use_cache=True).logits[:, -1, :]
            logits_group_list = []
            logits_group_list.append(logits_test.unsqueeze(0))
            
            for i in range(0,num_condition):
                logits_group = model(input_ids=input_ids_group[i], attention_mask=attention_mask_group[i], use_cache=True).logits[:, -1, :]
                logits_group_list.append(logits_group.unsqueeze(0))
            
            logits_group = torch.cat(logits_group_list, dim=0).to(device)

            logits_avg = logits_group.mean(dim=0).to(device)
            next_token = torch.argmax(logits_avg, dim=-1,keepdim=True).to(device)
            input_ids_test = torch.cat([input_ids_test, next_token], dim=-1).to(device)
            
            new_attention_mask = torch.ones((batch_size, 1), device=attention_mask_test.device).to(device)
            attention_mask_test = torch.cat([attention_mask_test, new_attention_mask], dim=-1).to(device)
            
            for i in range(0,num_condition):
                input_ids_group[i]=torch.cat([input_ids_group[i], next_token], dim=-1).to(device)
                attention_mask_group[i]=torch.cat([attention_mask_group[i], new_attention_mask], dim=-1).to(device)
            
            if (next_token == tokenizer.eos_token_id).all():
                break

            
        res = tokenizer.batch_decode(input_ids_test[:, pre_token:], skip_special_tokens=True)
        
        for j in range(0,len(res)):
            run_results[0].append({'text': res[j], 'label1': item[j]['label1'],'label2':item[j]['label2']})
            
    with open(output, 'w') as f:
        json.dump(run_results, f, ensure_ascii=False, indent=2)

    score, layer = compute_metric(output,task)
    acc.append(score)

    acc = np.array(acc)
    acc_mean = np.mean(acc)
    
    print("Multi CTG Acc:"+str(acc_mean))
    
    return run_results    
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='/data1/chh/models/meta-llama/Meta-Llama-3-8B-Instruct')

    # parser.add_argument('--model_path', type=str, default='/data1/chh/models/model_sft/llama3-8b/merge/qwen/sft3')
    parser.add_argument('--task', type=str, default='multi')
    parser.add_argument('--output', type=str, default='./results/batch_ctg/hs_test5.json')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_length', type=int, default=128)
    
    args = parser.parse_args()
    set_seed(args)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path,padding_side='left')
    model = LlamaForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16).to(device)

    tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    model = model.eval()
    CTG_hs(model,tokenizer,args.task,args.output)
    # CTG_logits(model,tokenizer,args.task,args.output,args.max_length)
    # CTG_hs_split(model,tokenizer,args.task,args.output)
    
    