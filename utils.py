import argparse
import json
import os
import random
import re
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import openai
import pandas as pd
import spacy
import torch
from googleapiclient import discovery
from scipy.special import expit, softmax
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

nlp = spacy.load('en_core_web_md')
device = torch.device("cuda")

topic_label = {
    'arts_&_culture':0, 'business_&_entrepreneurs':1, 'celebrity_&_pop_culture':2, 'diaries_&_daily_life':3, 'family':4, 
    'fashion_&_style':5, 'film_tv_&_video':6, 'fitness_&_health':7, 'food_&_dining':8, 'gaming':9,
    'learning_&_educational':10, 'music':11, 'news_&_social_concern':12, 'other_hobbies':13, 'relationships':14,
    'science_&_technology':15, 'sports':16, 'travel_&_adventure':17, 'youth_&_student_life':18
}
label_topic = {value: key for key, value in topic_label.items()}
label_sentiment1 = {
    0:'negative', 1:'neutral', 2:'positive'
}
label_sentiment2 = {
    0:'anger', 1:'disgust', 2:'fear', 3:'joy', 4:'neutral', 5:'sadness', 6:'surprise'
}

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
def get_data(file_path):
    testd_ata = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            item = json.loads(line)
            testd_ata.append(item)
    return testd_ata

def prompt_template(tokenizer,message,sys_prompt='Generate text according to the following instruction.') -> str:
    messages = [
    {"role": "system", "content":sys_prompt},
    {"role": "user", "content": message},
    ]
    
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
def process_test_datset(tokenizer,task,file_path):
    records=[]
    split_conditions = []
    
    test_df=get_data(file_path)
    
    for i in range(len(test_df)):
        test_message = "Instruction: {instruction}\n".format(instruction=test_df[i]['instruction'])
        prompt_test=prompt_template(tokenizer,test_message)
        temp={'prompt':prompt_test}
        
        for key, value in test_df[i].items():
            if key != 'instruction':
                temp[key] = value
        records.append(temp)
        
        if i==0:
            num_condition = sum(1 for key in temp.keys() if 'label' in key)
            
        
        for k,v in temp.items():
            if 'label' in k:
                prompt_condition="Instruction: Generate a text that fits the condition: {label}.\n".format(label=v.replace('_', ' ').replace('&', 'and'))
                prompt_condition=prompt_template(tokenizer,prompt_condition)                    
                split_conditions.append(prompt_condition)
    
    return records,split_conditions,num_condition

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

def classify_topic(device, model, tokenizer, text, label):
    '''
    refer to https://huggingface.co/cardiffnlp/tweet-topic-21-multi
    '''
    tokens = tokenizer(text, return_tensors='pt').to(device)
    if tokens.input_ids.shape[1] > 512:
        tokens.input_ids = tokens.input_ids[:, :512]
        tokens.attention_mask = tokens.attention_mask[:, :512]

    output = model(input_ids=tokens.input_ids, attention_mask=tokens.attention_mask)
    output = output[0][0].detach().cpu()
    scores = output.numpy()
    scores = expit(scores)
    pred = np.argmax(scores)
    predictions = (scores >= 0.5) * 1
    
    # return scores[topic_label[label]]
    return predictions[topic_label[label]]

    


def classify_sentiment(device, model1, model2, tokenizer1, tokenizer2, text, label):
    '''
    refer to https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest
    '''
    model, tokenizer = None, None
    index=0
    if label in ['negative', 'neutral', 'positive']:
        index=['negative', 'neutral', 'positive'].index(label)
        model = model1
        tokenizer = tokenizer1
    elif label in ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']:
        index=['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise'].index(label)
        model = model2
        tokenizer = tokenizer2
    encoded_input = tokenizer(text, return_tensors='pt').to(device)
    if encoded_input.input_ids.shape[1] > 512:
        encoded_input.input_ids = encoded_input.input_ids[:, :512]
        encoded_input.attention_mask = encoded_input.attention_mask[:, :512]

    # output = model(input_ids=encoded_input.input_ids, attention_mask=encoded_input.attention_mask)
    # output = output[0][0].detach().cpu()
    # scores = output.numpy()
    # scores = softmax(scores)

    # return scores[index]
    output = model(input_ids=encoded_input.input_ids, attention_mask=encoded_input.attention_mask)
    output = output[0][0].detach().cpu()
    scores = output.numpy()
    scores = softmax(scores)
    pred = np.argmax(scores)
    if len(scores) == 3:
        return label_sentiment1[pred] == label
    elif len(scores) == 7:
        return label_sentiment2[pred] == label

def compute_metric(output_filename,task):
    eval_models=load_eval_models(task)
    
    if task=='topic' and len(eval_models)!=2:
        exit()
    if task=='sentiment' and len(eval_models)!=4:
        exit()
    if task=='multi' and len(eval_models)!=6:
        exit()
        
    with open(output_filename, 'r') as f:
        run_results = json.load(f)
    
    max_acc = -1
    best_layer = 0
    acc_s = []
    for layer,pred_answers in run_results.items():
        acc = 0
        
        for pred in pred_answers:
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
        
        total_num = len(pred_answers)
        acc=acc*1.0/total_num
        print("ACC-%2s: %.4f" % (layer,acc))
            
        acc_s.append(acc)
        max_acc = max(acc,max_acc)
        if max_acc == acc:
            best_layer = layer
    print("ACC-MAX: %.4f\n" % (max_acc))
    if best_layer!=0:
        print("Best layer: {}\n".format(best_layer))
    log_file='./batch_ctg_log.txt'
    with open(log_file, 'a') as log_file:
        log_file.write("Output File: {}, Task: {}, ACC: {:.4f}, Best layer: {}\n".format(output_filename,task, max_acc,best_layer))
    return max_acc,best_layer


def extract_text(text):
    pattern = r'(?::\n\n|:\n)(.*)$'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1)
    else:
        return text
def get_lemma(word):
    word = word.lower()
    doc = nlp(word)
    lemmatized_word = doc[0].lemma_
    return lemmatized_word

def extract_words(sentence):
    words = re.findall(r'\w+(?:-\w+)*', sentence)
    words = [word.lower() for word in words]
    # convert all word to normal format
    words = [get_lemma(word) for word in words]
    return words

def detect_toxic(text):
    # Replace with your own Perspective API Key
    PERSPECTIVE_API_KEY = 'AIzaSyBAn6aHKU44iQ2rIf2bXO_k67y3FKDIofk'

    max_tries = 10
    num_tries = 0
    success = False
    response = None
    while num_tries < max_tries and not success:
        try:
            client = discovery.build(
                "commentanalyzer",
                "v1alpha1",
                developerKey=PERSPECTIVE_API_KEY,
                discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
                static_discovery=False,
            )
            analyze_request = {
                'comment': {'text': text},
                'requestedAttributes': {
                    'TOXICITY': {}
                }
            }
            response = client.comments().analyze(body=analyze_request).execute()
            success = True
        except Exception as e:
            print(f"Error encountered: {e}")
            if 'Attribute TOXICITY does not support request languages' in str(e) or 'COMMENT_EMPTY' in str(e):
                # If there is a situation where the language cannot be judged, 
                # it will be skipped directly and returned -1, without participating in the final judgment
                return -1

            num_tries += 1
            if num_tries == max_tries:
                print("Maximum number of tries reached. Aborting.")
                raise
            print(f"Retrying (attempt {num_tries}/{max_tries})...")
            time.sleep(10)

    return response['attributeScores']['TOXICITY']['summaryScore']['value']

