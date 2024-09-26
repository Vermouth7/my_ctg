import json
import os

os.environ["CUDA_VISIBLE_DEVICES"]='6'
import json
import os
import time
from http import HTTPStatus

import torch
from dashscope import Generation
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from utils import classify_sentiment, classify_topic

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

# Function to predict the label of generated text using a classification model
def predict_label(task,eval_models,text,label1,label2=None):
    if task == 'topic':
        score=classify_topic(device=device,model=eval_models[0],tokenizer=eval_models[1],text=text,label=label1)
    elif task=='sentiment':
        score=classify_sentiment(device=device, model1=eval_models[0], model2=eval_models[2], tokenizer1=eval_models[1], tokenizer2=eval_models[3],text=text, label=label1)
    elif task == 'multi':
        score1=classify_topic(device=device,model=eval_models[0],tokenizer=eval_models[1],text=text,label=label1)
        score2=classify_sentiment(device=device, model1=eval_models[2], model2=eval_models[4], tokenizer1=eval_models[3], tokenizer2=eval_models[5],text=text, label=label2)
        score = (score1 + score2 == 2)
    return score

# Function to generate text using an LLM API based on the provided instruction
def generate_text(instruction,condition):
    messages = [{'role': 'system', 'content': 'You are a skilled and imaginative writer.'},
                {'role': 'user', 'content': 'Please respond in English.\
                    Generate a text based on the following instructions: {}.\n \
                    While generating the text, make sure to incorporate vocabulary and semantics related to the following constraint: {}.\n \
                    Apart from adhering to this specific condition, feel free to generate creatively and randomly. \
                    Ensure the generated text does not exceed 100 words in length.'.format(instruction,condition)}]
    gen = Generation()

    max_tries = 15
    num_tries = 0
    success = False
    response = None
    while num_tries < max_tries and not success:
        try:
            response = gen.call(
                'qwen-plus',
                messages=messages,
                result_format='message'
            )
            if response.status_code == HTTPStatus.OK:
                response = response['output']['choices'][0]['message']['content'].strip()
                success = True
            else:
                print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
                    response.request_id, response.status_code,
                    response.code, response.message
                ))
                success = False
        except Exception as e:
            print(f"Error encountered: {e}")
            num_tries += 1
            if num_tries == max_tries:
                print("Maximum number of tries reached. Aborting.")
                raise
            print(f"Retrying (attempt {num_tries}/{max_tries})...")
            time.sleep(10)
    time.sleep(1)
    
    return response

def generate_text_multi(instruction,condition1,condition2):
    messages = [{'role': 'system', 'content': 'You are a skilled and imaginative writer.'},
                {'role': 'user', 'content': 'Please respond in English.\
                    Generate a text based on the following instructions: {}.\n \
                    While generating the text, make sure to incorporate vocabulary and semantics related to the both following constraint: {}, {}.\n \
                    Apart from adhering to these specific conditions, feel free to generate creatively and randomly. \
                    Ensure the generated text does not exceed 100 words in length.'.format(instruction,condition1,condition2)}]
    gen = Generation()

    max_tries = 10
    num_tries = 0
    success = False
    response = None
    while num_tries < max_tries and not success:
        try:
            response = gen.call(
                'qwen-plus',
                messages=messages,
                result_format='message'
            )
            if response.status_code == HTTPStatus.OK:
                response = response['output']['choices'][0]['message']['content'].strip()
                success = True
            else:
                print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
                    response.request_id, response.status_code,
                    response.code, response.message
                ))
                num_tries += 1
                success = False
        except Exception as e:
            print(f"Error encountered: {e}")
            num_tries += 1
            if num_tries == max_tries:
                print("Maximum number of tries reached. Aborting.")
                raise
            print(f"Retrying (attempt {num_tries}/{max_tries})...")
            time.sleep(10)
    time.sleep(1)
    return response


if __name__ == '__main__':
    task='multi'
    eval_models=load_eval_models(task)
    data=[]
    # Load the JSON file containing instructions and labels
    with open('/home/chh/repos/my_ctg/instructions/train/multi/multi_diversified.jsonl', 'r',encoding='utf-8') as file:
        for line in file:
            data_dict = json.loads(line.strip())
            data.append(data_dict)
    revised_data = []
    # Iterate through each instruction in the JSON data
    for item in tqdm(data, desc="Processing instructions"):
        try_times=0
        instruction = item['instruction']
        label1 = item['label1']
        label2= item['label2']
        # Generate text based on the instruction
        generated_text = generate_text_multi(instruction,label1,label2)
        print(generated_text)
        # Predict the label of the generated text
        predicted_label = predict_label(task,eval_models,generated_text,label1,label2)
        
        # Check if the predicted label matches the correct label
        while predicted_label != 1 and try_times<5:
            # If the label does not match, create a new instruction template for revision
            
            revised_instruction='Please reply in English. Re-generate a new content to avoid similarity with the previous one.\n \
                                Instruction: {}\n \
                                Pay attention to vocabulary and semantic accuracy to ensure that the new content has a higher relevance to the labels: "{}" and "{}".'.format(instruction,label1,label2)
            # Generate revised text based on the new instruction
            generated_text = generate_text_multi(revised_instruction,label1,label2)
            print(generated_text)
            predicted_label = predict_label(task,eval_models,generated_text,label1,label2)
            try_times+=1
        if predicted_label!=1:
            revised_data.append({
            'instruction': instruction,
            'output': generated_text,
            'label1': label1,
            'label2':label2,
            'revise':1
        })
        else:
            revised_data.append({
                'instruction': instruction,
                'output': generated_text,
                'label1': label1,
                'label2':label2,
            })

    # Optionally, save the revised dataset to a new JSON file
    with open('./sft/train_qwen/train_multi.json', 'w') as file:
        json.dump(revised_data, file, indent=4)
