import json
import os

os.environ["CUDA_VISIBLE_DEVICES"]='6'
import json
import os
import time
from http import HTTPStatus

import torch
from dashscope import Generation
from openai import OpenAI
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from utils import classify_sentiment, classify_topic

deepseek_api='sk-8733604a6ca649629cbee448ee90ad7b'

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


with open('/home/chh/repos/my_ctg/sft/train_qwen/train_multi.json','r',encoding='utf-8') as f:
    data=json.load(f)

eval_models=load_eval_models('multi')

client = OpenAI(api_key=deepseek_api, base_url="https://api.deepseek.com")
revised_data=[]

for item in tqdm(data, desc="Processing instructions"):
    if 'revise' in item:
        try_times=0
        
        response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are good at writing"},
            {"role": "user", "content": 'Please respond in English.\
                    Generate a text based on the following instructions: {}.\n \
                    While generating the text, make sure to incorporate vocabulary and semantics related to the both following constraint: {}, {}.\n \
                    Apart from adhering to these specific conditions, feel free to generate creatively and randomly. \
                    Ensure the generated text does not exceed 100 words in length.'.format(item['instruction'],item['label1'],item['label2'])},
        ],
        stream=False
        )
        generate_text=response.choices[0].message.content
        ans=predict_label('multi',eval_models,generate_text,item['label1'],item['label2'])
        while ans != 1 and try_times<5:
            revised_instruction='Please reply in English. Re-generate a new content to avoid similarity with the previous one.\n \
                                Instruction: {}\n \
                                Pay attention to vocabulary and semantic accuracy to ensure that the new content has a higher relevance to the labels: "{}" and "{}".'.format(item['instruction'],item['label1'],item['label2'])
            response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are good at writing"},
                {"role": "user", "content": revised_instruction},
            ],
            stream=False
            )
            try_times+=1
            generate_text=response.choices[0].message.content
            ans=predict_label('multi',eval_models,generate_text,item['label1'],item['label2'])
            
            
        if ans==1:
            item['output'] = generate_text

        print(ans)
        
with open('/home/chh/repos/my_ctg/sft/train_qwen/train_multi_temp.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)