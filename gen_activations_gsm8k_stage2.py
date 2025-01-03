import argparse
import json
import os

os.environ['CUDA_VISIBLE_DEVICES']='0'

import re
import time
from difflib import SequenceMatcher

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from batch_repe import repe_pipeline_registry
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import (AutoModel, AutoTokenizer, LlamaForCausalLM,
                          StoppingCriteria, StoppingCriteriaList, pipeline)
from utils import *
from vllm import LLM, SamplingParams

repe_pipeline_registry()
device = torch.device("cuda")

def nshot_chats(tokenizer,nshot_data: list, n: int, question: str) -> dict:

    def question_prompt(s):
        return f'Question: {s}'

    def answer_prompt(s):
        return f'Answer: {s}'

    chats = []
    random.seed(42)
    for qna in nshot_data[:n]:
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

def extract_substring(full_text, part_text):
    max_match_length = 0
    end_index = 0
    
    for i in range(len(part_text)):
        partial_text = part_text[i:]
        start_index = full_text.find(partial_text)
        
        if start_index != -1 and len(partial_text) > max_match_length:
            max_match_length = len(partial_text)
            end_index = start_index + max_match_length
    
    if max_match_length > 0:
        return full_text[:end_index]
    else:
        return None

def gen(args):
    model = LlamaForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16).to(device)
    model = model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path,padding_side='left')
    tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    run_results = []
    batch_size = 2
    test_data=[]
    prompt="You're a mathematician who's good at reasoning. Answer the following questions using detailed reasoning steps, and you MUST write the answer as an integer after '####'.\nQuestion: {question}"
    
    train_data=load_dataset("/data1/chh/datasets/openai/gsm8k",'main')
    train_data=train_data['train'].to_pandas().to_dict(orient='records')[:5000]
    with open(args.eval_file,'r') as test_file:
        test_data=json.load(test_file)
    test_data_2=[]
    for i in test_data:
        if i['result']==0:
            i['new_q']=nshot_chats(tokenizer,nshot_data=train_data, n=8, question=i['question']) + ' ' + i['add_inf']
            test_data_2.append(i)
                   
    inputs=[i['new_q'] for i in test_data_2]
    
    encodings = tokenizer(inputs, padding=True, truncation=True, return_tensors='pt').to(device)
    all_hidden_states = {}
    for i in tqdm(range(0, len(inputs), batch_size)):
        input_ids = encodings['input_ids'][i:i+batch_size]
        attention_mask = encodings['attention_mask'][i:i+batch_size]
        outputs = model.generate(input_ids, attention_mask=attention_mask,
                                 max_new_tokens=args.max_length,
                                 return_dict_in_generate=True,
                                 output_hidden_states=True)
        
        sample_hs=[]
        for token in range(len(outputs.hidden_states)):
            sample_hs.append(outputs.hidden_states[token][-1][:,-1])
        sample_hs=torch.stack(sample_hs,dim=0)
        sample_hs=sample_hs.transpose(0,1)
        sample_hs = sample_hs.cpu()
        for j, generated_output in enumerate(outputs.sequences):
            input_length = input_ids[j].size(0)
            
            generated_tokens = generated_output[input_length:]
            decoded_output = tokenizer.decode(generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            
            test_data_2[i + j]['generated_text'] = decoded_output
            all_hidden_states[i+j]=sample_hs[j]
            del test_data_2[i + j]['new_q']
    
    torch.save(all_hidden_states, args.original_data)
    with open(args.output_folder, 'w', encoding='utf-8') as output_file:
        json.dump(test_data_2, output_file, ensure_ascii=False, indent=4)


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
            record['result']=1
        else:
            record['result']=0

    accuracy = correct_count / total_count * 100
    print(f"Acc: {accuracy:.2f}%")
    with open(args.eval_file, 'w', encoding='utf-8') as output_file:
        json.dump(data, output_file, ensure_ascii=False, indent=4)
def comparison(args):
    sampling_params = SamplingParams(temperature=0.6, top_p=0.9,max_tokens=1024)
    model=LLM(model=args.model_path,gpu_memory_utilization=0.80)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path,padding_side='left')
    tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    inputs=[]
    data_com=[]
    with open(args.eval_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    with open('/home/chh/repos/my_ctg/instructions/template/compare_gsm8k_stage2.txt','r',encoding='utf-8') as f:
        template=f.read()
    for i in range(len(data)):
        record=data[i]
        record['id']=i
        data_com.append(record)
        if record['result']==0:
            content=record['key']            
            pattern = r'\{[^{}]*\}'
            match = re.findall(pattern, content)
            if match:
                content=match[0]
            try:
                if content.startswith("```json"): 
                    content = content[7:-3].strip()
                    answer = json.loads(content)
                else:
                    answer = json.loads(content)
            except Exception as e:
                    answer=None
            # print(answer)
            
            if answer:
                sentences=list(answer.values())
                inputs.append(prompt_template(tokenizer,template%(record['question'],record['answer'],record['generated_text'],sentences[0])))
            else:
                inputs.append(prompt_template(tokenizer,template%(record['question'],record['answer'],record['generated_text'],record['key'])))
                
            
    outputs = model.generate(inputs, sampling_params)
    res = [item.outputs[0].text for item in outputs]
    
    index=0
    for i in data_com:
        if i['result']==0:
            i['part']=res[index]
            index+=1
    with open(args.eval_file, 'w', encoding='utf-8') as output_file:
        json.dump(data_com, output_file, ensure_ascii=False, indent=4)

def find_best_match(sentence_tokens, text_tokens):
    best_match_start = 0
    best_match_length = 0
    best_match_ratio = 0
    
    for i in range(len(text_tokens) - len(sentence_tokens) + 1):
        window = text_tokens[i:i + len(sentence_tokens)]
        match_ratio = SequenceMatcher(None, sentence_tokens, window).ratio()
        
        if match_ratio > best_match_ratio:
            best_match_ratio = match_ratio
            best_match_start = i
            best_match_length = len(sentence_tokens)
    
    return best_match_start, best_match_start + best_match_length - 1, best_match_ratio

def find_token_pos(text, sentences, tokenizer):
    tokenized_text = tokenizer.encode(text, add_special_tokens=False)  
    positions = []
    
    for sentence in sentences:
        tokenized_sentence = tokenizer.encode(sentence, add_special_tokens=False)  
        
        start, end, match_ratio = find_best_match(tokenized_sentence, tokenized_text)
        positions.append((sentence, start, end, match_ratio))
    
    return positions

def extract_hs(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path,padding_side='left')
    tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    train_data=torch.load(args.original_data)
    with open(args.eval_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    positive_samples = []
    negative_samples = []
    
    for item in data:
        if item['result']==1:
            for i in range(0,train_data[item['id']].shape[0]):
                sample_data = train_data[item['id']][i]
                negative_samples.append((sample_data, 0))
            continue
        content=item['key']
        # json_start = content.find('{')
        # content=item['key'][json_start:]
        pattern = r'\{[^{}]*\}'
        match = re.findall(pattern, content)
        if match:
            content=match[0]
        try:
            if content.startswith("```json"): 
                content = content[7:-3].strip()
                answer = json.loads(content)
            else:
                answer = json.loads(content)
        except Exception as e:
                # print(f"json failed to parse: {e}")
                # print(f"content: {content}")
                answer=None
        # print(answer)
        
        if answer:
            sentences=list(answer.values())
            second_stage=[]
            second_stage.append(sentences[0])
            ref=item['generated_text']
            token_pos = find_token_pos(ref, second_stage, tokenizer)
            matched_positions = []
            for sentence, start, end, match_ratio in token_pos:
                for i in range(start, end):  
                    sample_data = train_data[item['id']][i]  
                    positive_samples.append((sample_data, 1))
                matched_positions.append((start, end))
            
    positive_count = len(positive_samples)
    print(positive_count)
    print(len(negative_samples))
    if len(negative_samples) > positive_count:
        negative_samples = random.sample(negative_samples, positive_count)
    
    dataset = positive_samples + negative_samples
    random.shuffle(dataset)
    
    inputs = [sample[0] for sample in dataset]
    labels = [sample[1] for sample in dataset]
    
    inputs_tensor = torch.stack(inputs)
    labels_tensor = torch.tensor(labels)
    
    torch.save((inputs_tensor, labels_tensor), args.classifier_data)
        
def sliding_window_mean(features, window_size=3):

    num_samples, feature_dim = features.shape
    new_features = [
        np.mean(features[i:i + window_size], axis=0)
        for i in range(num_samples - window_size + 1)
    ]
    return np.array(new_features)

def train_classifier_LR(args):
    data=torch.load(args.classifier_data)
    # data_stage1=torch.load('/data1/chh/my_ctg/classifier/gsm8k/train7.pt')
    neg_data=torch.load('/data1/chh/my_ctg/classifier/gsm8k/train_neg.pt')
    
    features1, labels1 = data
    # features2, labels2 = data_stage1
    neg_features, neg_labels = neg_data


    # features_combined = torch.cat((features1, features2), dim=0)
    # labels_combined = torch.cat((labels1, labels2), dim=0)

    
    # non_zero_indices = labels_combined != 0
    # features_combined = features_combined[non_zero_indices]
    # labels_combined = labels_combined[non_zero_indices]

    
    # label_1_count = (labels_combined == 1).sum()

    
    # num_new_neg = int(label_1_count * 0.5)

    
    # assert num_new_neg <= len(neg_features), "neg_data 的数据不足以满足需要"

    
    # new_neg_features = neg_features[:num_new_neg]
    # new_neg_labels = neg_labels[:num_new_neg]
    label_1_count = (labels1 == 1).sum().item()

    # 确定负样本需要的数量
    num_new_neg = label_1_count

    # 如果负样本数量不足，抛出错误；否则裁剪负样本
    assert num_new_neg <= len(neg_features), "neg_data 的数据不足以满足需要"
    new_neg_features = neg_features[:num_new_neg]
    new_neg_labels = neg_labels[:num_new_neg]
    features_combined = torch.cat((features1, new_neg_features), dim=0)
    labels_combined = torch.cat((labels1, new_neg_labels), dim=0)
    features_combined = features_combined.to(torch.float32).cpu().numpy()  # (num_samples, 4096)
    labels_combined = labels_combined.cpu().numpy()     
    # 打印结果统计
    print(f"最终标签为 0 的数据数量: {(labels_combined == 0).sum()}")
    print(f"最终标签为 1 的数据数量: {(labels_combined == 1).sum()}")

    # features_combined = sliding_window_mean(features_combined, window_size=3)
    # labels_combined = labels_combined[2:]
    
    X_train, X_test, y_train, y_test = train_test_split(features_combined, labels_combined, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000)  
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    joblib.dump(model, args.classifier)

class classifier_mlp(nn.Module):
    def __init__(self):
        super(classifier_mlp, self).__init__()
        self.fc1 = nn.Linear(4096, 2048)
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, 1) 

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  
        return x

def comparison_after_gen(args):
    sampling_params = SamplingParams(temperature=0.6, top_p=0.9,max_tokens=args.max_length)
    model=LLM(model=args.model_path,gpu_memory_utilization=0.80)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path,padding_side='left')
    tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    inputs=[]
    data_com=[]
    with open(args.eval_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    with open('/home/chh/repos/my_ctg/instructions/template/compare_gsm8k.txt','r',encoding='utf-8') as f:
        template=f.read()
    for i in range(len(data)):
        record=data[i]
        record['id']=i
        data_com.append(record)
        if record['result']==0:
            inputs.append(prompt_template(tokenizer,template%(record['question'],record['answer'],record['generated_text'])))
            
    outputs = model.generate(inputs, sampling_params)
    res = [item.outputs[0].text for item in outputs]
    
    index=0
    for i in data_com:
        if i['result']==0:
            i['key']=res[index]
            index+=1
    with open(args.eval_file, 'w', encoding='utf-8') as output_file:
        json.dump(data_com, output_file, ensure_ascii=False, indent=4)
def train_classifier_mlp(args):
    data = torch.load(args.classifier_data)
    features, labels = data  
    features = torch.tensor(features, dtype=torch.float32).to(device)
    labels = torch.tensor(labels, dtype=torch.float32).view(-1, 1).to(device)
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    model = classifier_mlp().to(device)
    criterion = nn.BCELoss()  
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    for epoch in tqdm(range(100)):  
        model.train()
        for batch_features, batch_labels in train_loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)  
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        X_test = X_test.to(device)
        y_pred = model(X_test).round()  
        accuracy = accuracy_score(y_test.cpu(), y_pred.cpu())
        print(f"Test Accuracy: {accuracy * 100:.2f}%")

    torch.save(model.state_dict(), args.classifier)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='/data1/chh/models/meta-llama/Meta-Llama-3-8B-Instruct')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--eval_file',type=str,default='./results/gsm8k_act/res2_stage2.json')
    parser.add_argument('--output_folder', type=str, default='./results/gsm8k_act/res2_stage2.json')
    parser.add_argument('--original_data',type=str,default='/data1/chh/my_ctg/classifier/gsm8k/demo2_stage2.pt')
    parser.add_argument('--classifier_data',type=str,default='/data1/chh/my_ctg/classifier/gsm8k/train_stage1.pt')
    parser.add_argument('--classifier',type=str,default='/data1/chh/my_ctg/classifier/gsm8k/logistic_regression_model10.pkl')
    
    
    args = parser.parse_args()
    set_seed(args)
    
    # comparison(args)
    # gen(args)
    # eval_func(args)
    # comparison_after_gen(args)
    # extract_hs(args)
    train_classifier_LR(args)
    # train_classifier_mlp(args) 
    
    