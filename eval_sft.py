import argparse
import json
import os

os.environ["CUDA_VISIBLE_DEVICES"]='6'
import re
import time

import numpy as np
import pandas as pd
import tensor_parallel as tp
import torch
import transformers
from tqdm import tqdm
from transformers import (AutoModelForCausalLM,
                          AutoModelForSequenceClassification, AutoTokenizer,
                          LlamaForCausalBatchICLLM, LlamaForCausalLM,
                          LlamaTokenizer)
from utils import (classify_sentiment, classify_topic, detect_toxic,
                   extract_text, extract_words, get_lemma, set_seed)

device = torch.device("cuda")


def load_eval_models(task):
    if task=='topic':
    # topic model
        MODEL1 = f"/data1/chh/models/cardiffnlp/tweet-topic-21-multi"
        eval_tokenizer1 = AutoTokenizer.from_pretrained(MODEL1)
        eval_model1 = AutoModelForSequenceClassification.from_pretrained(MODEL1)
        eval_model1.to(device)
        return [eval_model1,eval_tokenizer1]
    elif task=='sentiment':
        # sentiment model
        MODEL2 = f"/data1/chh/models/cardiffnlp/twitter-roberta-base-sentiment-latest"
        eval_tokenizer2 = AutoTokenizer.from_pretrained(MODEL2)
        eval_model2 = AutoModelForSequenceClassification.from_pretrained(MODEL2)
        eval_model2.to(device)
        # load models 2
        MODEL3 = f"/data1/chh/models/j-hartmann/emotion-english-roberta-large"
        eval_tokenizer3 = AutoTokenizer.from_pretrained(MODEL3)
        eval_model3 = AutoModelForSequenceClassification.from_pretrained(MODEL3)
        eval_model3.to(device)

        return [eval_model2,eval_tokenizer2,eval_model3,eval_tokenizer3]
    elif task=='multi':
        MODEL1 = f"/data1/chh/models/cardiffnlp/tweet-topic-21-multi"
        eval_tokenizer1 = AutoTokenizer.from_pretrained(MODEL1)
        eval_model1 = AutoModelForSequenceClassification.from_pretrained(MODEL1)
        eval_model1.to(device)
        MODEL2 = f"/data1/chh/models/cardiffnlp/twitter-roberta-base-sentiment-latest"
        eval_tokenizer2 = AutoTokenizer.from_pretrained(MODEL2)
        eval_model2 = AutoModelForSequenceClassification.from_pretrained(MODEL2)
        eval_model2.to(device)
        # load models 2
        MODEL3 = f"/data1/chh/models/j-hartmann/emotion-english-roberta-large"
        eval_tokenizer3 = AutoTokenizer.from_pretrained(MODEL3)
        eval_model3 = AutoModelForSequenceClassification.from_pretrained(MODEL3)
        eval_model3.to(device)
        return [eval_model1,eval_tokenizer1,eval_model2,eval_tokenizer2,eval_model3,eval_tokenizer3]
    else:
        return None
    
    
def compute_metric(output_filename,task,eval_models):
    if task=='topic' and len(eval_models)!=2:
        exit()
    if task=='sentiment' and len(eval_models)!=4:
        exit()
    if task=='multi' and len(eval_models)!=6:
        exit()
    with open(output_filename, 'r') as f:
        run_results = [json.loads(line) for line in f]
    
    acc = 0
    
    for record in run_results:
        if task == 'topic':
            score=classify_topic(device=device,model=eval_models[0],tokenizer=eval_models[1],text=record['text'],label=record['label'])
        elif task=='sentiment':
            score=classify_sentiment(device=device, model1=eval_models[0], model2=eval_models[2], tokenizer1=eval_models[1], tokenizer2=eval_models[3],text=record['text'], label=record['label'])
        elif task == 'multi':
            score1=classify_topic(device=device,model=eval_models[0],tokenizer=eval_models[1],text=record['text'],label=record['label1'])
            score2=classify_sentiment(device=device, model1=eval_models[2], model2=eval_models[4], tokenizer1=eval_models[3], tokenizer2=eval_models[5],text=record['text'], label=record['label2'])
            score = (score1 + score2 == 2)
        elif task=='detoxic':
            toxicity = detect_toxic(record['text'])
            score=1-toxicity
        acc += score
    
    total_num = len(run_results)
    acc=acc*1.0/total_num
        
    print("ACC: %.4f" % (acc))
    
    return acc


if __name__ == '__main__':
    task='sentiment'
    eval_file='/home/chh/repos/my_ctg/results_zero_shot/Meta-Llama-3-8B-Instruct_sft2_350/{}/generation.jsonl'.format(task)
    eval_models=load_eval_models(task)
    compute_metric(eval_file,task,eval_models)

