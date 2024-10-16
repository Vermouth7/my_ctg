import os

os.environ['CUDA_VISIBLE_DEVICES']='4'
import argparse
import json
import re
import time

import accelerate
import numpy as np
import sympy
import torch
import torch.nn.functional as F
from accelerate import Accelerator, PartialState
from answer_extraction import extract_math_answer
from batch_repe import repe_pipeline_registry
from datasets import load_dataset
from sympy.parsing.latex import parse_latex
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, LlamaForCausalLM, pipeline
from utils import *
from vllm import LLM, SamplingParams
from wrappedmodel import WrappedModel

repe_pipeline_registry()
device = torch.device("cuda")

def process_data(sample):
    res={}
    res['sub_ins']=[]
    res['sub_ins'].append(sample['instruction 1'])
    res['sub_ins'].append(sample['instruction 2'])
    
    return res

def get_split_hs(model,tokenizer,batch_data,batch_size):
    all_hiddens= []
    
    with open(os.path.join("/home/chh/repos/my_ctg/instructions/math/math_2steps_llama.json"), 'r', encoding='utf-8') as input_file:
        data=json.load(input_file)
    samples=[]
    for batch_group in batch_data:
        matched_samples = []
        for sample in batch_group:
            question = sample['problem']  
            
            matched_sample = next((item for item in data if item['problem'] == question), None)
            if matched_sample:
                matched_samples.append(matched_sample)
        samples.extend(matched_samples)
        
    for sample in accelerate.utils.tqdm(data, desc=f"Processing dataset", unit="line"):
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


def CTG_hs(args):
    distributed_state = PartialState()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path,padding_side='left')
    model = LlamaForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16,device_map=distributed_state.device)
    
    tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id    
    model = model.eval()
    model=WrappedModel(model,tokenizer)
    
    insert_layer=[18]
    layers=[i-1 for i in insert_layer]
    print('insert layer: ',insert_layer)
    
    run_results = []
    batch_size = 1
    
    # prompt="Answer the following questions, and you MUST present the final result in LaTeX using a \\boxed{{}} without any units.\nProblem: {problem}"
    # prompt="Answer the following questions, and you MUST put your final answer within \\boxed{{}}.\nProblem: {problem}"
    
    data=load_dataset("/data1/chh/datasets/lighteval/MATH")
    test_data=[]
    
    for i in data['test']:
        instruction=fewshot_samples(tokenizer, question=i['problem'])
        # instruction=prompt_template(tokenizer,prompt.format(problem=i['problem']))
        test_data.append({'temp':instruction,'problem':i['problem'],'solution':i['solution']})
    
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
            
            for r, i in zip(res, item):
                run_results.append({'problem': i['problem'], 'solution': i['solution'],'generated_text':r})
    torch.distributed.barrier()  
    res_gather = accelerate.utils.gather_object(run_results)
    torch.distributed.barrier()  
    
    if distributed_state.is_main_process:
        memo = set()
        final_res = []
        for result in res_gather:
            if result['problem'] not in memo:
                final_res.append(result)
                memo.add(result['problem'])
        with open(args.output_file, 'w', encoding='utf-8') as file:
            for item in final_res:
                file.write(json.dumps(item) + '\n')
        eval_func(args.output_file)

def vllm_gen(args):
    sampling_params = SamplingParams(temperature=0.6, top_p=0.95,max_tokens=args.max_new_tokens)
    model=LLM(model=args.model_path,gpu_memory_utilization=0.90)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path,padding_side='left')
    tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    run_results = []
    inputs=[]
    
    # prompt="Problem:{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
    # prompt="Answer the following questions, and you MUST put your final answer within \\boxed{{}}.\nProblem: {problem}"
    
    data=load_dataset("/data1/chh/datasets/lighteval/MATH")
    test_data=data['test']
    
    for i in test_data:
        # instruction=prompt_template(tokenizer,prompt.format(problem=i['problem']))
        # inputs.append(instruction)
        instruction=fewshot_samples(tokenizer, question=i['problem'])
        inputs.append(instruction)
        
    
    outputs = model.generate(inputs, sampling_params)
    res = [item.outputs[0].text for item in outputs]
        
    for r,i in zip(res,test_data):
        run_results.append({'problem':i['problem'],'solution':i['solution'],'generated_text': r})
    
    with open(args.output_file, 'w', encoding='utf-8') as output_file:
        json.dump(run_results, output_file, ensure_ascii=False, indent=4)
    eval_func(args.output_file)
def fewshot_samples(tokenizer,question) -> list[dict]:
    def question_prompt(s):
        return f'Problem: {s}'

    def answer_prompt(s):
        return f'Solution: {s}'
    nshot=[
        {
            "problem": "Find the domain of the expression  $\\frac{\\sqrt{x-2}}{\\sqrt{5-x}}$.}",
            "solution": "The expressions inside each square root must be non-negative. Therefore, $x-2 \\ge 0$, so $x\\ge2$, and $5 - x \\ge 0$, so $x \\le 5$. Also, the denominator cannot be equal to zero, so $5-x>0$, which gives $x<5$. Therefore, the domain of the expression is $\\boxed{[2,5)}$.\nFinal Answer: The final answer is $[2,5)$. I hope it is correct.",
        },
        {
            "problem": "If $\\det \\mathbf{A} = 2$ and $\\det \\mathbf{B} = 12,$ then find $\\det (\\mathbf{A} \\mathbf{B}).$",
            "solution": "We have that $\\det (\\mathbf{A} \\mathbf{B}) = (\\det \\mathbf{A})(\\det \\mathbf{B}) = (2)(12) = \\boxed{24}.$\nFinal Answer: The final answer is $24$. I hope it is correct.",
        },
        {
            "problem": "Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, how many times must Terrell lift them in order to lift the same total weight?",
            "solution": "If Terrell lifts two 20-pound weights 12 times, he lifts a total of $2\\cdot 12\\cdot20=480$ pounds of weight.  If he lifts two 15-pound weights instead for $n$ times, he will lift a total of $2\\cdot15\\cdot n=30n$ pounds of weight.  Equating this to 480 pounds, we can solve for $n$:\n\\begin{align*}\n30n&=480\\\n\\Rightarrow\\qquad n&=480/30=\\boxed{16}\n\\end{align*}\nFinal Answer: The final answer is $16$. I hope it is correct.",
        },
        {
            "problem": "If the system of equations\n\n\\begin{align*}\n6x-4y&=a,\\\n6y-9x  &=b.\n\\end{align*}has a solution $(x, y)$ where $x$ and $y$ are both nonzero,\nfind $\\frac{a}{b},$ assuming $b$ is nonzero.",
            "solution": "If we multiply the first equation by $-\\frac{3}{2}$, we obtain\n\n$$6y-9x=-\\frac{3}{2}a.$$Since we also know that $6y-9x=b$, we have\n\n$$-\\frac{3}{2}a=b\\Rightarrow\\frac{a}{b}=\\boxed{-\\frac{2}{3}}.$$\nFinal Answer: The final answer is $-\\frac{2}{3}$. I hope it is correct.",
        },
    ]
    chats = []

    for qna in nshot:
        chats.append(
            {"role": "user", "content": question_prompt(qna["problem"])})
        chats.append(
            {"role": "assistant", "content": answer_prompt(qna["solution"])})

    chats.append({"role": "user", "content": question_prompt(question)+"\nPlease reason step by step, and put your final answer within \\boxed{}."})
    return tokenizer.apply_chat_template(
        chats,
        tokenize=False,
        add_generation_prompt=True,
    )

def eval_func(eval_file):
    with open(eval_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    correct_count = 0
    total_count = len(data)

    for i, record in enumerate(data):
        answer = record.get('solution', '')
        reference = record.get('generated_text', '')
        question=record.get('problem','')
        
        answer_number = extract_math_answer(question,answer)
        reference_number = extract_math_answer(question,reference)

        if answer_number and reference_number and answer_number==reference_number:
            correct_count += 1

    accuracy = correct_count / total_count * 100
    print(f"Acc: {accuracy:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='/data1/chh/models/meta-llama/Meta-Llama-3-8B-Instruct')
    parser.add_argument('--output_file', type=str, default='./results/math/baseline_cot.json')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_new_tokens', type=int, default=512)
    parser.add_argument('--eval_file',type=str,default='./results/math/baseline_cot.json')
    
    args = parser.parse_args()
    set_seed(args)
    
    
    vllm_gen(args)
    # CTG_hs(args)
    
# lm_eval --model hf --model_args pretrained=/data1/chh/models/meta-llama/Meta-Llama-3-8B-Instruct, dtype="bfloat16" --tasks hendrycks_math --num_fewshot 4  --device cuda:0 --batch_size 4 --apply_chat_template --fewshot_as_multiturn
