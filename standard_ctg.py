import argparse
import json
import os

os.environ['CUDA_VISIBLE_DEVICES']='0'
import gc
import random

import numpy as np
import torch
from tqdm import tqdm
from transformers import (AutoConfig, AutoModel, AutoModelForCausalLM,
                          AutoModelForSequenceClassification, AutoTokenizer,
                          LlamaForCausalLM, LlamaTokenizer)
from utils import get_test_data, prompt_template, set_seed
from vllm import LLM, SamplingParams


def main(args):
    res=[]
    prompts=[]
    model = LLM(model=args.model_path,gpu_memory_utilization=0.95)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    sampling_params = SamplingParams(temperature=0.6, top_p=0.9,max_tokens=args.max_length)
    
    instructions = get_test_data(args.task)

    if args.mode == 'zero_shot':
        for i in instructions:
            tmp="Instruction: {}\n".format(i['instruction'])
            prompts.append(prompt_template(tokenizer,tmp))
    if args.mode == 'few_shot':
        # perform few shots
        few_shot_examples="1.Instruction: Can you write a happy text about family for me?\n \
            Response: Family is the heart of happiness, a sanctuary where love flows freely. Each member, like pieces of a puzzle, fits perfectly, creating a vibrant picture of support and joy. Home isn't just a place; it's the warmth felt within, surrounded by those who unconditionally love and uplift us.\n"
        for i in instructions:
            tmp=few_shot_examples+"Instruction: {}\nResponse:".format(i['instruction'])
            prompts.append(prompt_template(tokenizer,tmp))
    elif args.mode=='cot_zero_shot':
        for i in instructions:
            tmp="Let's think step by step. Instruction: {}\n ".format(i['instruction'])
            prompts.append(prompt_template(tokenizer,tmp))
    elif args.mode=='cot_few_shot':
        sys_prompt='You need to first think about the conditions that need to be met by this instruction, and when you are halfway through generating the text, output the text and indicate which conditions are met, and then continue generating until all conditions are met. You need to check your output during generation..'
        cot="Instruction: Can you write a happy text about family for me?" \
            "Conditions to be met: happy, family.\n" \
          "First Segment: \n" \
          "Family is the heart of happiness, where laughter fills every room.\n" \
          "Conditions already met: Happy,Family \n" \
          "Second Segment: \n" \
          "In the embrace of family, we find warmth and love that brighten our lives.\n" \
          "Check: \n" \
          "Happy: Warmth, love, and brighten our lives continue the positive tone.\n" \
          "Family: The concept of family is central, focusing on the emotional support it provides.\n" \
          "Complete text: \n" \
          "Family is the heart of happiness, where laughter fills every room. In the embrace of family, we find warmth and love that brighten our lives."
        for i in instructions:
            tmp=cot+"Instruction: {}\nResponse:".format(i['instruction'])
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
    parser.add_argument("--task", default='multi', type=str)
    parser.add_argument("--mode", default='zero_shot', type=str, choices=['zero_shot','few_shot','cot_zero_shot','cot_few_shot',])
    parser.add_argument("--output", default='./ctg/zero_shot.json', type=str)
    # decoding
    parser.add_argument("--top_p", default=0.9, type=float)
    parser.add_argument("--max_length", default=300, type=int)
    parser.add_argument("--temperature", default=0.6, type=float)
    args = parser.parse_args()
    set_seed(args)

    main(args)
    
