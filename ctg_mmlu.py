import os

os.environ['CUDA_VISIBLE_DEVICES']='0'
import argparse
import json
import re
import time

import numpy as np
import torch
import torch.nn.functional as F
from batch_repe import repe_pipeline_registry
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, LlamaForCausalLM, pipeline
from utils import *
from vllm import LLM, SamplingParams

repe_pipeline_registry()
device = torch.device("cuda")
options_prefix = ["A", "B", "C", "D"]

def process_data(sample):
    res={}
    res['sub_ins']=[]
    res['sub_ins'].append(sample['instruction 1'])
    res['sub_ins'].append(sample['instruction 2'])
    
    return res

def get_split_hs(model,tokenizer,):
    all_hiddens= []
    
    with open(os.path.join("/home/chh/repos/my_ctg/instructions/mmlu/mmlu_2steps_llama.json"), 'r', encoding='utf-8') as input_file:
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
    batch_size = 2
    test_data=[]
    split_hiddens= get_split_hs(model,tokenizer)
    prompt="The following are multiple choice questions about {label}.Your answer must include the correct option, such as (A),(B),(C) or (D).\nQuestion: {question}\nChoices: {choice}\nAnswer:"
    with open(os.path.join("/home/chh/repos/my_ctg/instructions/mmlu/mmlu_2steps_llama.json".format()), 'r', encoding='utf-8') as test_file:
        test_data = json.load(test_file)
    for i in test_data:
        choices_string=' '.join([f"({options_prefix[i]}) {choice}" for i, choice in enumerate(i['choices'])])
        i['new_q']=prompt_template(tokenizer,prompt.format(label=i['subject'],question=i['question'],choice=choices_string))
    
    ### COT
    # train_data=load_dataset("/data1/chh/datasets/openai/mmlu",'main')
    # train_data=train_data['train'].to_pandas().to_dict(orient='records')
    # for i in test_data:
    #     i['new_q']=nshot_chats(tokenizer,nshot_data=train_data, n=8, question=i['question'])
    
    batches_test = [test_data[i:i + batch_size] for i in range(0, len(test_data), batch_size)]
    batches_hidden=[split_hiddens[i:i + batch_size] for i in range(0,split_hiddens.shape[0], batch_size)]

    for index,item in enumerate(tqdm(batches_test, desc="Processing prompts")):
        inputs=[i['new_q'] for i in item]
        vector=batches_hidden[index]
        
        res = control_pipeline(inputs, activations=vector,token_pos=-1,batch_size=batch_size, max_new_tokens=args.max_length)
        
        res = [item[0]['generated_text'] for item in res]
        for r,i in zip(res,item):
            run_results.append({'question':i['question'],'answer':i['answer'],'generated_text':r})
    
    with open(args.output_folder, 'w', encoding='utf-8') as output_file:
        json.dump(run_results, output_file, ensure_ascii=False, indent=4)



    end_time = time.time()
    print("total run time %.2f" % (end_time - start_time))

def vllm_gen(args):
    sampling_params = SamplingParams(temperature=0.6, top_p=0.9,max_tokens=args.max_length)
    model=LLM(model=args.model_path,gpu_memory_utilization=0.90)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path,padding_side='left')
    tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    run_results = []
    batch_size = 1
    test_data=[]
    prompt="The following are multiple choice questions about {label}.Your answer must include only options without any other description, such as (A),(B),(C) or (D).\nQuestion: {question}\nChoices: {choice}\nAnswer:"
    with open(os.path.join("/home/chh/repos/my_ctg/instructions/mmlu/mmlu_2steps_llama.json".format()), 'r', encoding='utf-8') as test_file:
        test_data = json.load(test_file)
    # train_data = load_dataset("/data1/chh/datasets/lighteval/mmlu",'abstract_algebra')['test']
    # train_data=train_data['train'].to_pandas().to_dict(orient='records')

    for i in test_data:
        choices_string=' '.join([f"({options_prefix[i]}) {choice}" for i, choice in enumerate(i['choices'])])
        i['new_q']=prompt_template(tokenizer,prompt.format(label=i['subject'],question=i['question'],choice=choices_string))
        # i['new_q']=nshot_chats(tokenizer,nshot_data=train_data, n=8, question=i['question'])
    inputs=[i['new_q'] for i in test_data]
    
    outputs = model.generate(inputs, sampling_params)
    res = [item.outputs[0].text for item in outputs]
        
    for r,i in zip(res,test_data):
        run_results.append({'question':i['question'],'choices':i['choices'],'answer':i['answer'],'generated_text': r})
    
    with open(args.output_folder, 'w', encoding='utf-8') as output_file:
        json.dump(run_results, output_file, ensure_ascii=False, indent=4)

def nshot_chats(tokenizer,nshot_data: list, n: int, question: str) -> dict:

    def question_prompt(s):
        return f'Question: {s}'

    def answer_prompt(s):
        return f'Answer: {s}'

    chats = []

    random.seed(42)
    for qna in random.sample(nshot_data, n):
        chats.append(
            {"role": "user", "content": question_prompt(qna["question"])})
        chats.append(
            {"role": "assistant", "content": answer_prompt(qna["answer"])})

    chats.append({"role": "user", "content": question_prompt(question)+" Let's think step by step. At the end, you MUST write the answer as an integer after '####'."})

    return tokenizer.apply_chat_template(
        chats,
        tokenize=False,
        add_generation_prompt=True,
    )


def extract_ans_from_response(answer):
    for char in answer:
        if char in {'A', 'B', 'C', 'D'}:
            return char
    return None

def eval_func(args):
    with open(args.eval_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    correct_count = 0
    total_count = len(data)

    for i, record in enumerate(data):
        answer_idx = record.get('answer', '')
        reference = record.get('generated_text', '')

        answer_number = options_prefix[answer_idx]
        reference_number = extract_ans_from_response(reference)

        if answer_number == reference_number:
            correct_count += 1

    accuracy = correct_count / total_count * 100
    print(f"Acc: {accuracy:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='/data1/chh/models/meta-llama/Meta-Llama-3-8B-Instruct')

    # parser.add_argument('--model_path', type=str, default='/data1/chh/models/model_sft/llama3-8b/merge/qwen/sft3')
    parser.add_argument('--output_folder', type=str, default='./results/mmlu/res1.json')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_length', type=int, default=200)
    
    parser.add_argument('--eval_file',type=str,default='./results/mmlu/res1.json')
    
    args = parser.parse_args()
    set_seed(args)
    
    
    CTG_hs(args)
    # vllm_gen(args)
    eval_func(args)