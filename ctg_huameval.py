import os

os.environ['CUDA_VISIBLE_DEVICES']='0'
import argparse
import json
import re
import time

import numpy as np
import sympy
import torch
import torch.nn.functional as F
from answer_extraction import extract_math_answer
from batch_repe import repe_pipeline_registry
from datasets import load_dataset
from sympy.parsing.latex import parse_latex
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, LlamaForCausalLM, pipeline
from utils import *
from vllm import LLM, SamplingParams

repe_pipeline_registry()
device = torch.device("cuda")

def process_data(sample):
    res={}
    res['sub_ins']=[]
    res['sub_ins'].append(sample['instruction 1'])
    res['sub_ins'].append(sample['instruction 2'])
    
    return res

def get_split_hs(model,tokenizer,):
    all_hiddens= []
    
    with open(os.path.join("/home/chh/repos/my_ctg/instructions/gsm8k/gsm8k_2steps_llama_2.json"), 'r', encoding='utf-8') as input_file:
        data=json.load(input_file)
        
    for sample in tqdm(data, desc=f"Processing dataset", unit="line"):
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
    start_time = time.time()
        
    # best_layer = get_insert_layer(model,tokenizer,task,output)
    
    insert_layer=[18]

    print('insert layer: ',insert_layer)
    
    # print(logits.shape)
    # print(hiddens.shape)
    
    control_pipeline = pipeline(
        "ctg-control", 
        model=model, 
        tokenizer=tokenizer, 
        layers=insert_layer,
        device=device)
    
    
    
    run_results = []
    batch_size = 1
    test_data=[]
    split_hiddens= get_split_hs(model,tokenizer)
    # prompt="You're a mathematician who's good at reasoning. Answer the following questions using detailed reasoning steps, and you MUST write the answer as an integer after '####'.\nQuestion: {question}"
    prompt="You're a mathematician who's good at reasoning. Answer the following questions, and you MUST write the answer as an integer after '####'.\nQuestion: {question}"
    with open(os.path.join("/home/chh/repos/my_ctg/instructions/gsm8k/gsm8k_2steps_llama_2.json".format()), 'r', encoding='utf-8') as test_file:
        test_data = json.load(test_file)
    for i in test_data:
        i['new_q']=prompt_template(tokenizer,prompt.format(question=i['question']))
    
    ### COT
    # train_data=load_dataset("/data1/chh/datasets/openai/gsm8k",'main')
    # train_data=train_data['train'].to_pandas().to_dict(orient='records')
    # for i in test_data:
    #     i['new_q']=nshot_chats(tokenizer,nshot_data=train_data, n=8, question=i['question'])
    
    batches_test = [test_data[i:i + batch_size] for i in range(0, len(test_data), batch_size)]
    batches_hidden=[split_hiddens[i:i + batch_size] for i in range(0,split_hiddens.shape[0], batch_size)]

    for index,item in enumerate(tqdm(batches_test, desc="Processing prompts")):
        inputs=[i['new_q'] for i in item]
        vector=batches_hidden[index]
        res = control_pipeline(inputs,tokenizer, mode=2,activations=vector,token_pos=-1,batch_size=batch_size, max_new_tokens=args.max_length)
        
        res = [item[0]['generated_text'] for item in res]
        for r,i in zip(res,item):
            run_results.append({'question':i['question'],'answer':i['answer'],'generated_text':r})
    
    with open(args.output_folder, 'w', encoding='utf-8') as output_file:
        json.dump(run_results, output_file, ensure_ascii=False, indent=4)



    end_time = time.time()
    print("total run time %.2f" % (end_time - start_time))

def vllm_gen(args):
    sampling_params = SamplingParams(temperature=0.6, top_p=0.95,max_tokens=args.max_length)
    model=LLM(model=args.model_path,gpu_memory_utilization=0.90)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path,padding_side='left')
    tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    run_results = []
    inputs=[]
    
    prompt="Answer the following questions, and you MUST present the final result in LaTeX using a \\boxed{{}} without any units.\nProblem: {problem}"
    # prompt="Answer the following questions, and you MUST put your final answer within \\boxed{{}}.\nProblem: {problem}"
    prompt_nshot="{nshot}\nProblem: {problem}\nSolution: "
    
    nshot=list_fewshot_samples()
    data=load_dataset("/data1/chh/datasets/lighteval/MATH-Hard")
    train_data=data['train'].to_pandas().to_dict(orient='records')
    test_data=data['test']
    
    for i in test_data:
        instruction=prompt_template(tokenizer,prompt_nshot.format(nshot=nshot,problem=i['problem']))
        inputs.append(instruction)
        # i['new_q']=nshot_chats(tokenizer,nshot_data=train_data, n=8, question=i['question'])
    
    outputs = model.generate(inputs, sampling_params)
    res = [item.outputs[0].text for item in outputs]
        
    for r,i in zip(res,test_data):
        run_results.append({'problem':i['problem'],'solution':i['solution'],'generated_text': r})
    
    with open(args.output_folder, 'w', encoding='utf-8') as output_file:
        json.dump(run_results, output_file, ensure_ascii=False, indent=4)

def list_fewshot_samples() -> list[dict]:
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
            "problem": "If the system of equations\n\n\\begin{align*}\n6x-4y&=a,\\\n6y-9x &=b.\n\\end{align*}has a solution $(x, y)$ where $x$ and $y$ are both nonzero,\nfind $\\frac{a}{b},$ assuming $b$ is nonzero.",
            "solution": "If we multiply the first equation by $-\\frac{3}{2}$, we obtain\n\n$$6y-9x=-\\frac{3}{2}a.$$Since we also know that $6y-9x=b$, we have\n\n$$-\\frac{3}{2}a=b\\Rightarrow\\frac{a}{b}=\\boxed{-\\frac{2}{3}}.$$\nFinal Answer: The final answer is $-\\frac{2}{3}$. I hope it is correct.",
        },
    ]
    res=str()
    for i in nshot:
        string=f"Problem:\n{i['problem']}\nSolution:\n{i['solution']}\n"
        res+=string
    return res

def eval_func(args):
    with open(args.eval_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    correct_count = 0
    total_count = len(data)

    for i, record in enumerate(data):
        answer = record.get('solution', '')
        reference = record.get('generated_text', '')
        question=record.get('problem','')
        
        answer_number = extract_math_answer(question,answer)
        reference_number = extract_math_answer(question,reference)
        print('answer',answer_number)
        print('reference',reference_number)
        

        if answer_number and reference_number and answer_number==reference_number:
            correct_count += 1

    accuracy = correct_count / total_count * 100
    print(f"Acc: {accuracy:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='/data1/chh/models/meta-llama/Meta-Llama-3-8B-Instruct')

    # parser.add_argument('--model_path', type=str, default='/data1/chh/models/model_sft/llama3-8b/merge/qwen/sft3')
    parser.add_argument('--output_folder', type=str, default='./results/math/baseline.json')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_length', type=int, default=512)
    
    parser.add_argument('--eval_file',type=str,default='./results/math/baseline.json')
    
    args = parser.parse_args()
    set_seed(args)
    
    
    # CTG_hs(args)
    gen(args)
    eval_func(args)
    
    