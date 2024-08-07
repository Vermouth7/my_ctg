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

nlp = spacy.load('en_core_web_md')

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


def get_test_data(data_path):
    tasks = []
    with open(data_path, 'r') as f:
        for line in f.readlines():
            data = json.loads(line)
            tasks.append(data)
    return tasks


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


def get_result_dict(args, task, res):
    result_dict = None
    if args.task == 'multi':
        result_dict = {
            'instruction':task['instruction'], 'text':res, 'label1':task['label1'], 'label2':task['label2']
        }
    elif 'anti_label' in task.keys():
        result_dict = {
            'instruction':task['instruction'], 'text':res, 'label':task['label'], 'anti_label':task['anti_label']
        }
    elif args.task == 'detoxic':
        result_dict = {
            'instruction':task['instruction'], 'text':res, 'context_id':task['context_id']
        }
    else:
        result_dict = {
            'instruction':task['instruction'], 'text':res, 'label':task['label']
        }
    return result_dict

