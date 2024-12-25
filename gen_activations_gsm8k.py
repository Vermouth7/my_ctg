import argparse
import json
import os

os.environ['CUDA_VISIBLE_DEVICES']='2'

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


def gen(args):
    model = LlamaForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16).to(device)
    model = model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path,padding_side='left')
    tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    run_results = []
    batch_size = 4
    test_data=[]
    prompt='Given the following problem, reason and give a final answer to the problem.\nProblem:{}\nYour response should end with "The final answer is [answer]" where [answer] is the response to the problem.'
    
    # train_data=load_dataset("/data1/chh/datasets/openai/gsm8k",'main')
    # train_data=train_data['train'].to_pandas().to_dict(orient='records')[:5000]
    with open('/home/chh/repos/my_ctg/results/gsm8k_act/train_wrong.json','r',encoding='utf-8') as f:
        train_data=json.load(f)
    for i in train_data:
        i['new_q']=prompt_template(tokenizer,prompt.format(i['question']))
        # i['new_q']=nshot_chats(tokenizer,nshot_data=train_data, n=8, question=i['question'])
        
    inputs=[i['new_q'] for i in train_data]
    
    encodings = tokenizer(inputs, padding=True, truncation=True, return_tensors='pt').to(device)
    all_hidden_states = {}
    for i in tqdm(range(0, len(inputs), batch_size)):
        input_ids = encodings['input_ids'][i:i+batch_size]
        attention_mask = encodings['attention_mask'][i:i+batch_size]
        outputs = model.generate(input_ids, attention_mask=attention_mask,
                                 max_new_tokens=512,
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
            
            train_data[i + j]['generated_text'] = decoded_output
            all_hidden_states[i+j]=sample_hs[j]
            del train_data[i + j]['new_q']
    
    torch.save(all_hidden_states, args.original_data)
    with open(args.output_folder, 'w', encoding='utf-8') as output_file:
        json.dump(train_data, output_file, ensure_ascii=False, indent=4)


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
def extract_final_answer(text):
    """
    Extracts the numeric answer from the text that starts with 'The final answer is'.
    :param text: Input text containing the calculation and final answer.
    :return: The extracted numeric answer as a float or integer.
    """
    match = re.search(r'The final answer is[\s\$]*([\d.,]+)\.?', text)  # Added optional trailing period
    if match:
        # Clean the number (remove commas if any) and convert it to float or int
        number = match.group(1).replace(',', '')
        return number
    return None  # Return None if no match found
def eval_func(args):
    with open(args.eval_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    correct_count = 0
    total_count = len(data)

    for i, record in enumerate(data):
        answer = record.get('answer', '')
        reference = record.get('generated_text', '')

        answer_number = extract_ans_from_response(answer)
        reference_number = extract_final_answer(reference)
        if reference_number:
            if str(answer_number) == reference_number:
                correct_count += 1
                record['result']=1
            else:
                record['result']=0
        else:
            record['result']=9

    accuracy = correct_count / total_count * 100
    print(f"Acc: {accuracy:.2f}%")
    with open(args.eval_file, 'w', encoding='utf-8') as output_file:
        json.dump(data, output_file, ensure_ascii=False, indent=4)
def comparison(args):
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
            # for i in range(0,train_data[item['id']].shape[0]):
            #     sample_data = train_data[item['id']][i]
            #     negative_samples.append((sample_data, 0))
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
            first_stage=[]
            first_stage.append(sentences[0])
            
            ref=item['generated_text']
            token_pos = find_token_pos(ref, first_stage, tokenizer)
            matched_positions = []
            for sentence, start, end, match_ratio in token_pos:
                for i in range(start, end):  
                    sample_data = train_data[item['id']][i]  
                    positive_samples.append((sample_data, 1))
                matched_positions.append((start, end))
            
            # current_pos = 0
            # for start, end in matched_positions:
            #     if current_pos < start:
            #         for i in range(current_pos, start):
            #             sample_data = train_data[item['id']][i]
            #             negative_samples.append((sample_data, 0))
            #     current_pos = end
                
            # if current_pos < train_data[item['id']].shape[0]:
            #     for i in range(current_pos, train_data[item['id']].shape[0]):
            #         sample_data = train_data[item['id']][i]
            #         negative_samples.append((sample_data, 0))
    positive_count = len(positive_samples)
    print(positive_count)
    # print(len(negative_samples))
    # if len(negative_samples) > positive_count:
    #     negative_samples = random.sample(negative_samples, positive_count)
    
    # dataset = positive_samples + negative_samples
    # random.shuffle(dataset)
    dataset = positive_samples
    
    inputs = [sample[0] for sample in dataset]
    labels = [sample[1] for sample in dataset]
    
    inputs_tensor = torch.stack(inputs)
    labels_tensor = torch.tensor(labels)
    
    torch.save((inputs_tensor, labels_tensor), args.classifier_data)
        
def extract_hs_correct(args):
    train_data=torch.load(args.original_data)
    negative_samples=[]
    data_ids=[52, 136, 564, 612, 836, 900, 984, 1216, 1380, 1460, 1648, 1696, 1916, 2120, 2176, 2236, 2444, 2556, 2692, 3008, 3244, 3848, 4024, 4368, 4552, 4636, 4720, 4892, 5392, 5444, 5536, 5952, 5976, 6028, 6220, 6224, 6228, 6232, 6236, 6244, 6252, 4988, 6096, 181, 777, 849, 857, 977, 1417, 1661, 1973, 2193, 2277, 2337, 2349, 2481, 2645, 3293, 3337, 3345, 3437, 3537, 3645, 3841, 3853, 4025, 4337, 4473, 4569, 5073, 5089, 5233, 5313, 5405, 5633, 5809, 6077, 6197, 6221, 6225, 6229, 6233, 6237, 6241, 6245, 6, 174, 222, 250, 638, 1090, 1218, 1438, 1446, 1514, 1594, 1754, 1862, 1898, 1922, 2022, 2462, 2522, 2610, 2722, 2754, 2758, 2770, 3346, 3362, 3558, 3842, 3970, 4282, 4486, 4618, 5130, 5134, 5198, 5326, 5478, 5506, 5518, 5638, 5886, 6030, 6042, 6102, 6182, 6222, 6226, 6230, 6234, 6246, 6250, 207, 411, 559, 679, 967, 1035, 1103, 1335, 1959, 2175, 2211, 2271, 2399, 2427, 2459, 2519, 2615, 3067, 3079, 3155, 3211, 3499, 3735, 3759, 3779, 3847, 4159, 4355, 4395, 4543, 4823, 4831, 5187, 5411, 6003, 6159, 6179, 6219, 6223, 6227, 6231, 6235, 6239, 6247, 6251]
    for item in data_ids:
        
        for i in range(0,train_data[item].shape[0]):
            sample_data = train_data[item][i]
            negative_samples.append((sample_data, 0))

    print(len(negative_samples))
    inputs = [sample[0] for sample in negative_samples]
    labels = [sample[1] for sample in negative_samples]
    
    inputs_tensor = torch.stack(inputs)
    labels_tensor = torch.tensor(labels)
    
    torch.save((inputs_tensor, labels_tensor), args.classifier_data)
    
def train_classifier_LR(args):
    data=torch.load(args.classifier_data)
    features, labels = data  

    features = features.to(torch.float32).cpu().numpy()  # (num_samples, 4096)
    labels = labels.cpu().numpy()      # (num_samples,)

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

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
    parser.add_argument('--eval_file',type=str,default='./results/gsm8k_act/res_wrong.json')
    parser.add_argument('--output_folder', type=str, default='./results/gsm8k_act/res_wrong.json')
    parser.add_argument('--original_data',type=str,default='/data1/chh/my_ctg/classifier/gsm8k/demo_wrong.pt')
    parser.add_argument('--classifier_data',type=str,default='/data1/chh/my_ctg/classifier/gsm8k/train_stage1.pt')
    parser.add_argument('--classifier',type=str,default='/data1/chh/my_ctg/classifier/gsm8k/logistic_regression_model11.pkl')
    
    
    args = parser.parse_args()
    set_seed(args)
    
    # gen(args)
    # eval_func(args)
    # comparison(args)
    extract_hs(args)
    # extract_hs_correct(args)
    # train_classifier_LR(args)
    # train_classifier_mlp(args) 
    
    