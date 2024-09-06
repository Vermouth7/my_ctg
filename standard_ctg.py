import argparse
import json
import os

os.environ['CUDA_VISIBLE_DEVICES']='6'
import gc
import random

import numpy as np
import torch
from tqdm import tqdm
from transformers import (AutoConfig, AutoModel, AutoModelForCausalLM,
                          AutoModelForSequenceClassification, AutoTokenizer,
                          LlamaForCausalLM, LlamaTokenizer)
from utils import get_data, prompt_template, set_seed
from vllm import LLM, SamplingParams


def main(args):
    res=[]
    prompts=[]
    model = LLM(model=args.model_path,gpu_memory_utilization=0.95)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    sampling_params = SamplingParams(temperature=0.6, top_p=0.9,max_tokens=args.max_length)
    
    instructions = get_data('/home/chh/repos/my_ctg/instructions/test/{}_lite.jsonl'.format(args.task))

    if args.mode == 'zero_shot':
        for i in instructions:
            tmp="Instruction: {}\n".format(i['instruction'])
            prompts.append(prompt_template(tokenizer,tmp))
    if args.mode == 'few_shot':
        # perform few shots
        # few_shot_examples="1.Instruction: Can you write a happy text about family for me?\n \
        #     Response: Family is where love begins. In the warmth of our family, happiness blooms like a garden in spring.\n \
        #       2. Instruction: Gimme a text about fashion or style, but make it angry.\n  \
        #           Response: Fashion today is a confusing battleground. It's frustrating to see creativity stifled by the relentless push for conformity.\n \
        #               3. Instruction: Gimme a text about fashion or style, but make it angry.\n \
        #                   Response: The thrill of travel ignites a spark within us. Every destination brings a wave of joy, turning ordinary moments into extraordinary memories.\n"
        few_shot_examples="1. Instruction: Generate a positive text centered on travel or adventure\n \
        Response:  Embrace the thrill of adventure as you voyage through uncharted territories, where each step unveils breathtaking vistas and heartwarming encounters. Travel is a canvas painted with colors of joy, learning, and self-discovery. Every new place is a page in the book of life, filled with stories waiting to be told. Roam through ancient cities, hike majestic mountains, and sail serene seas. Let every journey enrich your soul and expand your horizons, for the world is vast and brimming with wonders waiting to inspire and delight.\n \
        2. Instruction: Generate a fearful text centered on culture:\n \
        Response: In the shadowed alleys of forgotten culture, where ancient arts whisper through decaying theaters, a malevolent presence lurks. Painted faces of porcelain masks leer from dimly lit stalls, their eyes following you with unsettling interest. Statues weep blood-red tears that stain crumbling frescoes, and the air hums with an eerie melody, a song of despair and dread. Here, the past clings to the present, binding them in a chilling embrace of fear.\n \
        3. Instruction: Generate a surprising text centered around relationships:\n \
        Response: In a whirlwind of unexpected events, Alex discovered an old, cryptic letter hidden in the attic, revealing a shocking family secret: their great-aunt was actually their great-uncle. This revelation reshaped family relationships, introducing surprises and new connections. Bonds strengthened as they navigated this surprise together, embracing the truth and each other.\n"
        for i in instructions:
            tmp=few_shot_examples+"Instruction: {}\nResponse:".format(i['instruction'])
            prompts.append(prompt_template(tokenizer,tmp))
    elif args.mode=='cot_zero_shot':
        for i in instructions:
            tmp="Let's think step by step. Instruction: {}\n ".format(i['instruction'])
            prompts.append(prompt_template(tokenizer,tmp))
    elif args.mode=='cot_few_shot':
        sys_prompt='Generate text that matches the conditions based on the input instruction. You need to check all the conditions to be met at the beginning of the generation and check if some of them are met during the generation process. Finally output the complete text. Your output formatting should match the 3 examples below.'
        
        example_1 = (
            "1. Instruction: Can you write a happy text about family for me?\n"
            "Conditions to be met: happy, family.\n"
            "First Segment: Family is where love begins.\n"
            "Conditions already met: Family.\n"
            "Second Segment: In the warmth of our family, happiness blooms like a garden in spring.\n"
            "Conditions already met: Family, Happy.\n"
            "Complete text: Family is where love begins. In the warmth of our family, happiness blooms like a garden in spring.\n"
        )

        example_2 = (
            "2. Instruction: Gimme a text about fashion or style, but make it angry.\n"
            "Conditions to be met: fashion or style, angry.\n"
            "First Segment: Fashion today is a confusing battleground.\n"
            "Conditions already met: Fashion.\n"
            "Second Segment: It's frustrating to see creativity stifled by the relentless push for conformity.\n"
            "Conditions already met: Fashion, Angry.\n"
            "Complete text: Fashion today is a confusing battleground. It's frustrating to see creativity stifled by the relentless push for conformity.\n"
        )

        example_3 = (
            "3. Instruction: Joyful text with a travel focus needed:\n"
            "Conditions to be met: joyful, travel.\n"
            "First Segment: The thrill of travel ignites a spark within us.\n"
            "Conditions already met: Travel.\n"
            "Second Segment: Every destination brings a wave of joy, turning ordinary moments into extraordinary memories.\n"
            "Conditions already met: Travel, Joyful.\n"
            "Complete text: The thrill of travel ignites a spark within us. Every destination brings a wave of joy, turning ordinary moments into extraordinary memories.\n"
        )

        for i in instructions:
            tmp=example_1+example_2+example_3+"Instruction: {}\n".format(i['instruction'])
            prompts.append(prompt_template(tokenizer,tmp,sys_prompt=sys_prompt))
        
    
    outputs = model.generate(prompts, sampling_params)
    # Print the outputs.
    if args.task=='multi':
        for ins,output in zip(instructions,outputs):
            generated_text = output.outputs[0].text
            tmp = {'text':generated_text,'label1':ins['label1'],'label2':ins['label2']}
            res.append(tmp)
    else:
        for ins,output in zip(instructions,outputs):
            generated_text = output.outputs[0].text
            tmp = {'text':generated_text,'label':ins['label']}
            res.append(tmp)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=4)


    
    

    


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--model_path", default='/data1/chh/models/meta-llama/Meta-Llama-3-8B-Instruct', type=str)
    parser.add_argument('--output_folder', type=str, default='./results/followbench/baseline')
    
    # parser.add_argument("--task", default='multi', type=str)
    # parser.add_argument("--mode", default='few_shot', type=str, choices=['zero_shot','few_shot','cot_zero_shot','cot_few_shot',])
    # parser.add_argument("--output", default='./results/standard/few_shot.json', type=str)
    # decoding
    parser.add_argument("--top_p", default=0.9, type=float)
    parser.add_argument("--max_length", default=1024, type=int)
    parser.add_argument("--temperature", default=0.6, type=float)
    args = parser.parse_args()
    set_seed(args)
    
    constraint_types=['content', 'situation', 'style', 'format', 'example', 'mixed']
    model = LLM(model=args.model_path,gpu_memory_utilization=0.95)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    sampling_params = SamplingParams(temperature=0.6, top_p=0.9,max_tokens=args.max_length)
    
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
