import argparse
import json
import os

os.environ['CUDA_VISIBLE_DEVICES']='1'
import gc
import random

import numpy as np
import torch
from tqdm import tqdm
from transformers import (AutoConfig, AutoModel, AutoModelForCausalLM,
                          AutoModelForSequenceClassification, AutoTokenizer,
                          LlamaForCausalLM, LlamaTokenizer)
from utils import (classify_sentiment, classify_topic, detect_toxic,
                   load_eval_models, set_seed)
from vllm import LLM, SamplingParams

device=torch.device('cuda')

def compute_metric_standard(input_filename,task):
    eval_models=load_eval_models(task)
    
    if task=='topic' and len(eval_models)!=2:
        exit()
    if task=='sentiment' and len(eval_models)!=4:
        exit()
    if task=='multi' and len(eval_models)!=6:
        exit()
        
    with open(input_filename, 'r') as f:
        run_results = json.load(f)
    
    acc = 0
    
    for pred in tqdm(run_results,desc='Processing evaluation'):
        if task == 'topic':
            score=classify_topic(device=device,model=eval_models[0],tokenizer=eval_models[1],text=pred['text'],label=pred['label'])
        elif task=='sentiment':
            score=classify_sentiment(device=device, model1=eval_models[0], model2=eval_models[2], tokenizer1=eval_models[1], tokenizer2=eval_models[3],text=pred['text'], label=pred['label'])
        elif task == 'multi':
            score1=classify_topic(device=device,model=eval_models[0],tokenizer=eval_models[1],text=pred['text'],label=pred['label1'])
            score2=classify_sentiment(device=device, model1=eval_models[2], model2=eval_models[4], tokenizer1=eval_models[3], tokenizer2=eval_models[5],text=pred['text'], label=pred['label2'])
            score = (score1 + score2 == 2)
        elif task=='detoxic':
            toxicity = detect_toxic(pred)
            score=1-toxicity
        acc += score
    
    total_num = len(run_results)
    acc=acc*1.0/total_num
        
    with open(args.log_file, 'a') as log_file:
        log_file.write("Input File: {}, Task: {}, ACC: {:.4f}\n".format(args.input_file,task, acc))
    
    return acc

    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--task", default='multi', type=str)
    parser.add_argument("--input_file", default='./results/standard/few_shot.json', type=str)
    parser.add_argument('--log_file', type=str, default='./log.txt')
    
    args = parser.parse_args()
    set_seed(args)

    acc=compute_metric_standard(args.input_file,args.task)
    
