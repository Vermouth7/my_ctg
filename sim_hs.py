import os

os.environ['CUDA_VISIBLE_DEVICES']='4'
import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from batch_repe import repe_pipeline_registry
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, LlamaForCausalLM, pipeline
from utils import compute_metric, process_test_datset, set_seed

repe_pipeline_registry()
device = torch.device("cuda")

def get_condition_output(model,tokenizer,prompts,num_condition,pos):
    all_hiddens= []
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
        

        all_hiddens.append(mean_hidden_states)
    hiddens = torch.cat(all_hiddens, dim=0)

    return hiddens

def get_test_output(model,tokenizer,dataset):
    batch_size=8
    
    batches = [dataset[i:i + batch_size] for i in range(0, len(dataset), batch_size)]
    all_hiddens=[]
    
    for batch in tqdm(batches):
        encoded_inputs = tokenizer(batch, return_tensors="pt",padding=True)
        encoded_inputs.to(device)
        with torch.no_grad():
            outputs=model(**encoded_inputs,output_hidden_states=True,return_dict=True)
            hidden_states= outputs.hidden_states
            
        stacked_hidden_states = torch.stack([layer_output[:, -1:, :] for layer_output in hidden_states])
        stacked_hidden_states = torch.transpose(stacked_hidden_states, 0, 1)
        all_hiddens.append(stacked_hidden_states)
    hiddens=torch.cat(all_hiddens,dim=0)
    
    return hiddens

def compare_hs(model,tokenizer):
    layer_similarities=[]
    
    test_data,split,num_condition=process_test_datset(tokenizer,'multi','/home/chh/repos/my_ctg/instructions/val/multi_val.jsonl')
    dataset=[i['prompt'] for i in test_data]
    
    merge_hs=get_condition_output(model=model,tokenizer=tokenizer,prompts=split,num_condition=num_condition,pos=-1)
    test_hs=get_test_output(model=model,tokenizer=tokenizer,dataset=dataset)
    for layer in range(merge_hs.size(1)):
        merge_layer_vectors = merge_hs[:, layer, 0, :]
        test_layer_vectors = test_hs[:, layer, 0, :]
    
        cos_sim_matrix = F.cosine_similarity(merge_layer_vectors, test_layer_vectors, dim=-1)
        
        mean_sim = cos_sim_matrix.mean().item()
        layer_similarities.append(mean_sim)
        plt.figure(figsize=(10, 6))
        plt.plot(range(0, len(layer_similarities) + 0), layer_similarities, marker='o')

        plt.title('Average Similarity per Layer')
        plt.xlabel('Layer')
        plt.ylabel('Average Similarity')
        plt.grid(True)

        plt.savefig('./average_similarity_per_layer3.png')
    for i, sim in enumerate(layer_similarities):
        print(f"Layer {i} average similarity: {sim}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--model_path', type=str, default='/data1/chh/models/meta-llama/Meta-Llama-3-8B-Instruct')
    parser.add_argument('--model_path', type=str, default='/data1/chh/models/model_sft/llama3-8b/merge/qwen/sft3')
    

    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    set_seed(args)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path,padding_side='left')
    model = LlamaForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16).to(device)

    tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    model = model.eval()
    
    compare_hs(model,tokenizer)