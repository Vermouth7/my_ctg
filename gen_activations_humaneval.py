import argparse
import json
import os

os.environ['CUDA_VISIBLE_DEVICES']='7'

import re
import time
from difflib import SequenceMatcher

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from answer_extraction import extract_math_answer
from batch_repe import repe_pipeline_registry
from datasets import load_dataset
from human_eval.data import read_problems, write_jsonl
from MBPP.human_eval.evaluation import evaluate_functional_correctness
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
    
def extract_code_blocks(text):
    code_blocks = re.findall(f'```python\n(.*?)```', text, re.DOTALL)
    if code_blocks:
        return code_blocks[0]
    else:
        return text
def convert_for_evaluation(example):
    gpt_completion = example['gpt_completion']
    generation = gpt_completion
    try:
        code_block: str = re.findall(f'```python\n(.*?)```', gpt_completion, re.DOTALL | re.IGNORECASE)[0]
        generation = code_block
    except Exception as ex:
        print("Failed to extract codeblock:\n{}".format(gpt_completion))

    example['generation'] = generation
    return example

def read_test_examples(data_path: str):
    def format_test_example(q, tests, code: str=None):
        prompt = ">>> Problem:\n{}\n>>> Test Cases:\n{}\n".format(q.strip(), "\n".join(tests))
        if code:
            code = code.replace("\r", "").replace("\t", "    ")
            prompt += "\n>>> Code:\n```python\n{}\n```".format(code)
        return prompt

    examples = [json.loads(x) for x in open(data_path)]
    print("Read all {} examples from {} over!".format(len(examples), data_path))

    # test_cases
    examples_str = []
    for i in range(1,4):
        ex = examples[i]
        q, test, code = ex['text'], ex['test'], ex['code']
        ex_prompt = format_test_example(q, test, code)
        example_prompt = '- Example {}:\n{}'.format(i, ex_prompt)
        examples_str += [example_prompt]

    for i in range(0, len(examples)):
        ex = examples[i]
        q, test, code = ex['text'], ex['test'], ex['code']
        
        prompt = format_test_example(q, test, code=None)

        prompt_with_shots = '''
Please refer the given examples and generate a python function for my problem.
Examples are listed as follows:
{}

Here is my problem:
{}
'''.strip().format('\n\n'.join(examples_str), prompt)
        yield {
            'task_id': ex['task_id'],
            'prompt': prompt_with_shots
        }

def gen(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path,padding_side='left')
    model = LlamaForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16).to(device)
    
    tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    model = model.eval()
    
    
    run_results = []
    batch_size = 1
    
    # problems = [json.loads(x) for x in open('/home/chh/repos/my_ctg/MBPP/data/mbpp.jsonl')]
    problems=list(read_test_examples('/home/chh/repos/my_ctg/MBPP/data/mbpp.jsonl'))
    train_data=[]
    format_tabs=True
    
    for i in problems:
        prompt=prompt_template(tokenizer,i['prompt'])
        train_data.append(dict(task_id=i['task_id'],prompt=prompt))
        
    
    batches_test = [train_data[i:i + batch_size] for i in range(0, len(train_data), batch_size)]
    all_hidden_states = {}

    for index,item in enumerate(tqdm(batches_test, desc="Processing prompts")):
        inputs=[i['prompt'] for i in item]
        inputs = tokenizer(inputs, return_tensors="pt",padding=True).to(model.model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=0.1,
            top_p=0.95,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,  
            output_hidden_states=True,
            return_dict_in_generate=True,
        )
        sample_hs=[]
        
        for token in range(len(outputs.hidden_states)):
            sample_hs.append(outputs.hidden_states[token][-1][:,-1])
        sample_hs=torch.stack(sample_hs,dim=0)
        sample_hs=sample_hs.transpose(0,1)
        sample_hs = sample_hs.cpu()
        
        
        for j, generated_output in enumerate(outputs.sequences):
            input_length = inputs.input_ids[j].size(0)
            
            generated_tokens = generated_output[input_length:]
            decoded_output = tokenizer.decode(generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            
            train_data[index + j]['generation'] = extract_code_blocks(decoded_output)
            all_hidden_states[index+j]=sample_hs[j]
    for i in train_data:
        del i['prompt']
    torch.save(all_hidden_states, args.original_data)
    
    write_jsonl(args.output_file, train_data)

def instruct_prompt(prompt: str) -> str:
    return f"""Below is an instruction that describes a task about python code. Write a response that only contains python code to complete the request.\n\n### Instruction:\n{prompt}\n\n### Response:"""


def comparison(args):
    sampling_params = SamplingParams(temperature=0.6, top_p=0.9,max_tokens=args.max_new_tokens)
    model=LLM(model=args.model_path,gpu_memory_utilization=0.80)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path,padding_side='left')
    tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    inputs=[]
    data_com=[]
    with open('/home/chh/repos/my_ctg/results/humaneval_act/res1.json_results.jsonl', 'r', encoding='utf-8') as f:
        data=[json.loads(line) for line in f]
    with open('/home/chh/repos/my_ctg/instructions/template/compare_mbpp.txt','r',encoding='utf-8') as f:
        template=f.read()
    with open('/home/chh/repos/my_ctg/MBPP/data/mbpp.jsonl', 'r', encoding='utf-8') as f:
        original_data=[json.loads(line) for line in f]
    for i in range(len(data)):
        record=data[i]
        record['id']=i
        data_com.append(record)
        if record['passed']==False:
            for item in original_data:
                if record['task_id'] == item['task_id']:
                    inputs.append(prompt_template(tokenizer,template.format(prompt=item['text'],answer=item['code'],generated=record['generation'])))
    
    outputs = model.generate(inputs, sampling_params)
    res = [item.outputs[0].text for item in outputs]
    
    index=0
    for i in data_com:
        if i['passed']==False:
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
        if item['passed']==True:
        #     for i in range(0, len(train_data[item['id']])):
        #         sample_data = train_data[item['id']][i]
        #         positive_samples.append((sample_data, 1))
        # else:
        #     for i in range(0, len(train_data[item['id']])):
        #         sample_data = train_data[item['id']][i]
        #         positive_samples.append((sample_data, 1))
            continue
        content=item['key']
        json_start = content.find('{')
        content=item['key'][json_start:]
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
                
        if answer:
            sentences=list(answer.values())
            ref=item['generation']
            token_pos = find_token_pos(ref, sentences, tokenizer)
            matched_positions = []
            for sentence, start, end, match_ratio in token_pos:
                for i in range(start, end):  
                    sample_data = train_data[item['id']][i]  
                    positive_samples.append((sample_data, 1))
                matched_positions.append((start, end))
            
            current_pos = 0
            for start, end in matched_positions:
                if current_pos < start:
                    for i in range(current_pos, start):
                        sample_data = train_data[item['id']][i]
                        negative_samples.append((sample_data, 0))
                current_pos = end
                
            if current_pos < train_data[item['id']].shape[0]:
                for i in range(current_pos, train_data[item['id']].shape[0]):
                    sample_data = train_data[item['id']][i]
                    negative_samples.append((sample_data, 0))
        
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
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.1, random_state=42)

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

    # parser.add_argument('--model_path', type=str, default='/data1/chh/models/model_sft/llama3-8b/merge/qwen/sft3')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_new_tokens', type=int, default=512)
    parser.add_argument('--eval_file',type=str,default='./results/humaneval_act/res1.json')
    parser.add_argument('--output_file', type=str, default='./results/humaneval_act/res1.json')
    parser.add_argument('--original_data',type=str,default='/home/chh/repos/my_ctg/sft/classifier/humaneval/demo.pt')
    parser.add_argument('--classifier_data',type=str,default='/home/chh/repos/my_ctg/sft/classifier/humaneval/train.pt')
    parser.add_argument('--classifier',type=str,default='/home/chh/repos/my_ctg/sft/classifier/humaneval/logistic_regression_model.pkl')
    
    
    args = parser.parse_args()
    set_seed(args)
    
    # gen(args)
    # comparison(args)
    # extract_hs(args)
    train_classifier_LR(args)
    # train_classifier_mlp(args)
    
    