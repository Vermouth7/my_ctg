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
    
    with open(os.path.join("/home/chh/repos/my_ctg/instructions/gsm8k/gsm8k_2steps_llama_5.json"), 'r', encoding='utf-8') as input_file:
        data=json.load(input_file)
        
    samples=[]
    for batch_group in batch_data:
        matched_samples = []
        for sample in batch_group:
            question = sample['question']  
            
            matched_sample = next((item for item in data if item['question'] == question), None)
            if matched_sample:
                matched_samples.append(matched_sample)
        samples.extend(matched_samples)
    for sample in accelerate.utils.tqdm(samples, desc=f"Processing dataset", unit="line"):
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
    batches_hidden=[all_hiddens[i:i + batch_size] for i in range(0,all_hiddens.shape[0], batch_size)]
    return batches_hidden


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
    
    run_results = []
    batch_size = 1
    test_data=[]
    # prompt="You're a mathematician who's good at reasoning. Answer the following questions using detailed reasoning steps, and you MUST write the answer as an integer after '####'.\nQuestion: {question}"
    # prompt="You're a mathematician who's good at reasoning. Answer the following questions, and you MUST write the answer as an integer after '####'.\nQuestion: {question}"
    
    with open(os.path.join("/home/chh/repos/my_ctg/instructions/gsm8k/gsm8k_2steps_llama.json".format()), 'r', encoding='utf-8') as test_file:
        test_data = json.load(test_file)
    test_data=test_data
    # for i in test_data:
    #     i['temp']=prompt_template(tokenizer,prompt.format(question=i['question']))
    
    ### COT
    train_data=load_dataset("/data1/chh/datasets/openai/gsm8k",'main')
    train_data=train_data['train'].to_pandas().to_dict(orient='records')
    for i in test_data:
        i['temp']=nshot_chats(tokenizer,nshot_data=train_data, n=8, question=i['question'])
    
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
                mode=1,
                output_hidden_states=True
            )
            
            res = tokenizer.batch_decode(
                [ids[input_ids_cutoff:] for ids in generated_ids],
                skip_special_tokens=True,
            )
            
            for r, i in zip(res, item):
                run_results.append({'question': i['question'], 'answer': i['answer'],'generated_text':r})
    
    distributed_state.wait_for_everyone()
    res_gather = accelerate.utils.gather_object(run_results)
    
    if distributed_state.is_main_process:
        memo = set()
        final_res = []
        for result in res_gather:
            if result['question'] not in memo:
                final_res.append(result)
                memo.add(result['question'])
        with open(args.output_file, 'w', encoding='utf-8') as file:
            json.dump(final_res, file, ensure_ascii=False, indent=4)
        eval_func(args)

def vllm_gen(args):
    sampling_params = SamplingParams(temperature=0.6, top_p=0.9,max_tokens=args.max_new_tokens)
    model=LLM(model=args.model_path,gpu_memory_utilization=0.90)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path,padding_side='left')
    tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    run_results = []
    test_data=[]
    # prompt="You're a mathematician who's good at reasoning. Answer the following questions using detailed reasoning steps, and you MUST write the answer as an integer after '####'.\nQuestion: {question}"
    prompt="You're a mathematician who's good at reasoning. Answer the following questions, and you MUST write the answer as an integer after '####'.\nQuestion: {question}"
    
    with open(os.path.join("/home/chh/repos/my_ctg/instructions/gsm8k/gsm8k_2steps_llama_5_label.json".format()), 'r', encoding='utf-8') as test_file:
        test_data = json.load(test_file)
    train_data=load_dataset("/data1/chh/datasets/openai/gsm8k",'main')
    train_data=train_data['train'].to_pandas().to_dict(orient='records')
    for i in test_data:
        
        # i['temp']=prompt_template(tokenizer,prompt.format(question=i['question']))
        # i['temp']=prompt_template(tokenizer,prompt.format(question=i['question']+i['instruction 1']+i['instruction 2']))
        i['temp']=nshot_chats(tokenizer,nshot_data=train_data, n=8, question=i['question']+i['instruction 1']+i['instruction 2'])
    inputs=[i['temp'] for i in test_data]
    
    outputs = model.generate(inputs, sampling_params)
    res = [item.outputs[0].text for item in outputs]
        
    for r,i in zip(res,test_data):
        run_results.append({'question':i['question'],'answer':i['answer'],'generated_text': r})
    
    with open(args.output_file, 'w', encoding='utf-8') as output_file:
        json.dump(run_results, output_file, ensure_ascii=False, indent=4)
    eval_func(args)
def nshot_chats(tokenizer,nshot_data: list, n: int, question: str) -> dict:

    def question_prompt(s):
        return f'Question: {s}'

    def answer_prompt(s):
        return f'Answer: {s}'

    chats = []
    random.seed(42)
    shot_data=[{'question': "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? <THOUGHT>Identify the number of clips sold in May by calculating half of the 48 clips sold in April. Sum the number of clips sold in April and May to determine the total number of clips sold.<\THOUGHT>", 
               'answer': "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72"}, 
               {'question': "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn? <THOUGHT>Calculate Weng's earnings based on the time worked in hours, recognizing that 50 minutes is a fraction of an hour. Multiply Weng's hourly rate of $12 by the fraction of the hour that corresponds to 50 minutes to determine her total earnings.<\THOUGHT>",
                'answer': "Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute.\nWorking 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10.\n#### 10"}, 
               {'question': "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet? <THOUGHT>Determine how much money Betty currently has by adding half of the wallet's cost, the $15 from her parents, and twice that amount from her grandparents. Subtract the total amount Betty has from the wallet's cost of $100 to find how much more money she needs.<\THOUGHT>", 
                'answer': "In the beginning, Betty has only 100 / 2 = $<<100/2=50>>50.\nBetty's grandparents gave her 15 * 2 = $<<15*2=30>>30.\nThis means, Betty needs 100 - 50 - 30 - 15 = $<<100-50-30-15=5>>5 more.\n#### 5"}, 
               {'question': "Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read? <THOUGHT> Calculate the total number of pages Julie has read so far by adding the pages read yesterday and today, where today is twice the amount read yesterday. Determine the number of remaining pages by subtracting the total pages read from the total pages in the book, then calculate half of the remaining pages for her reading goal tomorrow.<\THOUGHT>",
                'answer': "Maila read 12 x 2 = <<12*2=24>>24 pages today.\nSo she was able to read a total of 12 + 24 = <<12+24=36>>36 pages since yesterday.\nThere are 120 - 36 = <<120-36=84>>84 pages left to be read.\nSince she wants to read half of the remaining pages tomorrow, then she should read 84/2 = <<84/2=42>>42 pages.\n#### 42"},
               {'question': "James writes a 3-page letter to 2 different friends twice a week.  How many pages does he write a year? <THOUGHT>Determine how many pages James writes per week by multiplying the number of pages per letter (3) by the number of letters (2), and then by the number of times he writes per week (2). Multiply the total number of pages James writes per week by the number of weeks in a year (52) to find the total number of pages written in a year.<\THOUGHT>", 
                'answer': "He writes each friend 3*2=<<3*2=6>>6 pages a week\nSo he writes 6*2=<<6*2=12>>12 pages every week\nThat means he writes 12*52=<<12*52=624>>624 pages a year\n#### 624"}, 
               {'question': "Mark has a garden with flowers. He planted plants of three different colors in it. Ten of them are yellow, and there are 80% more of those in purple. There are only 25% as many green flowers as there are yellow and purple flowers. How many flowers does Mark have in his garden? <THOUGHT>Determine the number of purple flowers by calculating 80% more than the 10 yellow flowers, and then find the total number of yellow and purple flowers combined. Calculate the number of green flowers as 25% of the total number of yellow and purple flowers, and then sum all the flowers (yellow, purple, and green) to find the total number of flowers in the garden.<\THOUGHT>", 
                'answer': "There are 80/100 * 10 = <<80/100*10=8>>8 more purple flowers than yellow flowers.\nSo in Mark's garden, there are 10 + 8 = <<10+8=18>>18 purple flowers.\nPurple and yellow flowers sum up to 10 + 18 = <<10+18=28>>28 flowers.\nThat means in Mark's garden there are 25/100 * 28 = <<25/100*28=7>>7 green flowers.\nSo in total Mark has 28 + 7 = <<28+7=35>>35 plants in his garden.\n#### 35"}, 
               {'question': "Albert is wondering how much pizza he can eat in one day. He buys 2 large pizzas and 2 small pizzas. A large pizza has 16 slices and a small pizza has 8 slices. If he eats it all, how many pieces does he eat that day? <THOUGHT>Determine how many slices of pizza Albert gets from the two large pizzas by multiplying the number of slices per large pizza by 2. Determine how many slices Albert gets from the two small pizzas by multiplying the number of slices per small pizza by 2, and then sum the results to get the total number of pieces he eats.<\THOUGHT>", 
                'answer': "He eats 32 from the largest pizzas because 2 x 16 = <<2*16=32>>32\nHe eats 16 from the small pizza because 2 x 8 = <<2*8=16>>16\nHe eats 48 pieces because 32 + 16 = <<32+16=48>>48\n#### 48"}, 
               {'question': "Ken created a care package to send to his brother, who was away at boarding school.  Ken placed a box on a scale, and then he poured into the box enough jelly beans to bring the weight to 2 pounds.  Then, he added enough brownies to cause the weight to triple.  Next, he added another 2 pounds of jelly beans.  And finally, he added enough gummy worms to double the weight once again.  What was the final weight of the box of goodies, in pounds? <THOUGHT>Determine the initial weight after Ken adds jelly beans to reach 2 pounds, which serves as the starting weight for subsequent additions. Calculate the weight after each addition (tripling with brownies, adding jelly beans, and then doubling with gummy worms) to find the final weight of the box.<\THOUGHT>", 
                'answer': "To the initial 2 pounds of jelly beans, he added enough brownies to cause the weight to triple, bringing the weight to 2*3=<<2*3=6>>6 pounds.\nNext, he added another 2 pounds of jelly beans, bringing the weight to 6+2=<<6+2=8>>8 pounds.\nAnd finally, he added enough gummy worms to double the weight once again, to a final weight of 8*2=<<8*2=16>>16 pounds.\n#### 16"}]
    for qna in shot_data[:n]:
        chats.append(
            {"role": "user", "content": question_prompt(qna["question"])})
        chats.append(
            {"role": "assistant", "content": answer_prompt(qna["answer"])})

    chats.append({"role": "user", "content": "Let's think step by step. At the end, you MUST write the answer as an integer after '####'\n."+question_prompt(question)})
    
    return tokenizer.apply_chat_template(
        chats,
        tokenize=False,
        add_generation_prompt=True,
    )


def extract_ans_from_response(answer: str, eos=None):
    if eos:
        answer = answer.split(eos)[0].strip()

    answer = answer.split('####')[-1].strip()

    for remove_char in [',', '$', '%', 'g']:
        answer = answer.replace(remove_char, '')

    try:
        return int(answer)
    except ValueError:
        return answer
def eval_func(args):
    with open(args.eval_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    correct_count = 0
    total_count = len(data)

    for i, record in enumerate(data):
        answer = record.get('answer', '')
        reference = record.get('generated_text', '')

        answer_number = extract_ans_from_response(answer)
        reference_number = extract_ans_from_response(reference)

        if answer_number == reference_number:
            correct_count += 1

    accuracy = correct_count / total_count * 100
    print(f"Acc: {accuracy:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='/data1/chh/models/meta-llama/Meta-Llama-3-8B-Instruct')
    parser.add_argument('--output_file', type=str, default='./results/gsm8k/patch1_label.json')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_new_tokens', type=int, default=512)
    parser.add_argument('--eval_file',type=str,default='./results/gsm8k/patch1_label.json')
    
    args = parser.parse_args()
    set_seed(args)
    
    
    # CTG_hs(args)
    vllm_gen(args)