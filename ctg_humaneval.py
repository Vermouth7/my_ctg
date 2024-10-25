import os

os.environ['CUDA_VISIBLE_DEVICES']='7'
import argparse
import copy
import json
import re
import time

import numpy as np
import sympy
import torch
import torch.nn.functional as F
from batch_repe import repe_pipeline_registry
from datasets import load_dataset
from human_eval.data import read_problems, write_jsonl
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, LlamaForCausalLM, pipeline
from utils import *
from wrappedmodel import WrappedModel

repe_pipeline_registry()
device = torch.device("cuda")

def process_data(sample):
    res={}
    res['sub_ins']=[]
    res['sub_ins'].append(sample['instruction 1'])
    res['sub_ins'].append(sample['instruction 2'])
    
    return res

def get_split_hs(model,tokenizer,num_samples_per_task):
    all_hiddens= []
    
    with open(os.path.join("/home/chh/repos/my_ctg/instructions/humaneval/humaneval_2steps_llama_2.json"), 'r', encoding='utf-8') as input_file:
        data=json.load(input_file)
    expanded_data=[]
    for item in data:
        expanded_data.extend([item] * num_samples_per_task)  
    for sample in tqdm(expanded_data, desc=f"Processing dataset", unit="line"):
        res=process_data(sample)
        
        hidden_states_list = []
        for sub_instruction in res['sub_ins']:
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


def CTG_hs(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path,padding_side='left')
    model = LlamaForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16).to(device)
    
    tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    model = model.eval()
    model=WrappedModel(model,tokenizer)
    
    insert_layer=[18]
    layers=[i-1 for i in insert_layer]
    print('insert layer: ',insert_layer)
    
    num_samples_per_task = 10
    run_results = []
    batch_size = 1
    split_hiddens= get_split_hs(model,tokenizer,num_samples_per_task)
    
    problems = read_problems()
    test_data=[]
    format_tabs=True
    
    for task_id in problems:
        for _ in range(num_samples_per_task):
            if format_tabs:
                prompt = problems[task_id]["prompt"].replace("    ", "\t")
            else:
                prompt = problems[task_id]["prompt"]
            prompt=prompt_template(tokenizer,build_deepseekcoder_instruction(prompt))
            test_data.append(dict(task_id=task_id,prompt=prompt))
        
    
    batches_test = [test_data[i:i + batch_size] for i in range(0, len(test_data), batch_size)]
    batches_hidden=[split_hiddens[i:i + batch_size] for i in range(0,split_hiddens.shape[0], batch_size)]

    for index,item in enumerate(tqdm(batches_test, desc="Processing prompts")):
        inputs=[i['prompt'] for i in item]
        vector=batches_hidden[index]
        
        model.reset()
        model.set_controller(layer_ids=layers,activations=vector)
        model.set_pos(inputs)
        inputs = tokenizer(inputs, return_tensors="pt",padding=True).to(model.model.device)
        input_ids_cutoff = inputs.input_ids.size(dim=1)

        generated_ids = model.generate(
            **inputs,
            use_cache=False,
            do_sample=True,
            max_new_tokens=args.max_new_tokens,
            temperature=0.1,
            top_p=0.95,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,  
            mode=0,
            output_hidden_states=True
        )

        res = tokenizer.batch_decode(
            [ids[input_ids_cutoff:] for ids in generated_ids],
            skip_special_tokens=True,
        )
        res=[extract_code_blocks(completion) for completion in res]
        for r,i in zip(res,item):
            result = dict(
                task_id=i['task_id'],
                completion=r,
            )
            run_results += [result]
    
    
    write_jsonl(args.output_file, run_results)


def filter_code(completion: str) -> str:
    # The program tends to overwrite, we only take the first function
    completion = completion.lstrip("\n")
    return completion.split("\n\n")[0]


def fix_indents(text: str) -> str:
    return text.replace("\t", "    ")
def extract_code_blocks(text):
    code_blocks = re.findall(r'```python(.*?)```', text, re.DOTALL)
    if code_blocks:
        return code_blocks[0]
    else:
        return text

@torch.inference_mode()
def generate_batch_completion(
    model, tokenizer, prompt, batch_size
) -> list[str]:
    input_batch = [prompt for _ in range(batch_size)]
    inputs = tokenizer(input_batch, return_tensors="pt").to(model.device)
    input_ids_cutoff = inputs.input_ids.size(dim=1)

    generated_ids = model.generate(
        **inputs,
        use_cache=True,
        max_new_tokens=args.max_new_tokens,
        temperature=0.1,
        top_p=0.95,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,  # model has no pad token
        mode=0
    )

    batch_completions = tokenizer.batch_decode(
        [ids[input_ids_cutoff:] for ids in generated_ids],
        skip_special_tokens=True,
    )

    return [extract_code_blocks(completion) for completion in batch_completions]


def gen(args):
    model = LlamaForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16).eval().to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path,padding_side='left')
    tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    
    problems = read_problems()
    num_samples_per_task = 10
    samples = []
    pbar = tqdm(total=len(problems) * num_samples_per_task)
    format_tabs=True
    
    for task_id in problems:
        if format_tabs:
            prompt = problems[task_id]["prompt"].replace("    ", "\t")
        else:
            prompt = problems[task_id]["prompt"]
        prompt=prompt_template(tokenizer,build_deepseekcoder_instruction(prompt))
        batch_completions = generate_batch_completion(
            model, tokenizer, prompt, num_samples_per_task
        )

        for sample in batch_completions:
            result = dict(
                task_id=task_id,
                completion=sample,
            )

            samples += [result]

        pbar.update(num_samples_per_task)

    write_jsonl(args.output_file, samples)

def instruct_prompt(prompt: str) -> str:
    return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nComplete the following Python code without any tests or explanation\n{prompt}\n\n### Response:"""
def build_deepseekcoder_instruction(question,languge='python'):
    return '''
Please continue to complete the function. You are not allowed to modify the given code and do the completion only. Please return all completed function in a codeblock. Here is the given code to do completion:
```{}
{}
```
'''.strip().format(languge.lower(), question.strip())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='/data1/chh/models/meta-llama/Meta-Llama-3-8B-Instruct')
    parser.add_argument('--output_file', type=str, default='./results/humaneval/res5.json')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_new_tokens', type=int, default=512)
    
    args = parser.parse_args()
    set_seed(args)
    
    CTG_hs(args)
    # gen(args)
    
        


# evaluate_functional_correctness ./results/humaneval/baseline.json