import argparse
import os

os.environ['CUDA_VISIBLE_DEVICES']='7'
import json
import time

import numpy as np
import pandas as pd
import torch
from batch_repe import repe_pipeline_registry
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, LlamaForCausalLM, pipeline
from utils import *

repe_pipeline_registry()

choices = ["A", "B", "C", "D"]
device = torch.device("cuda")

def process_data(sample):
    res={}
    res['sub_ins']=[]
    res['sub_ins'].append(sample['instruction 1'])
    res['sub_ins'].append(sample['instruction 2'])
    return res
def get_split_hs(model,tokenizer,file_path):
    all_hiddens= []
    print(file_path)
    with open(file_path, 'r', encoding='utf-8') as input_file:
        data=json.load(input_file)
        
    for sample in tqdm(data, desc=f"Processing dataset", unit="line"):
        res=process_data(sample)
        hidden_states_list = []
        for sub_instruction in res['sub_ins']:
            sub_instruction=prompt_template(tokenizer=tokenizer,message=sub_instruction)
            inputs = tokenizer(sub_instruction, return_tensors='pt')
            inputs.to(device)
            with torch.no_grad():
                outputs = model(**inputs,output_hidden_states=True)
            
            hidden_states = outputs.hidden_states
            stacked_hidden_states = torch.stack([layer_output[:, -1:, :] for layer_output in hidden_states]) # 33 1 token_pos 4096
            
            # stacked_hidden_states = torch.mean(stacked_hidden_states, dim=2, keepdim=True)
            stacked_hidden_states = torch.transpose(stacked_hidden_states, 0, 1)
            hidden_states_list.append(stacked_hidden_states)

        hidden_states_tensor = torch.stack(hidden_states_list)
        average_hidden_state = torch.mean(hidden_states_tensor, dim=0)
        average_hidden_state = average_hidden_state.squeeze(0)
        all_hiddens.append(average_hidden_state)
    all_hiddens=torch.stack(all_hiddens) # data 33 4096
    return all_hiddens


def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    for j in range(4):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, 1][j])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, 2])
    return prompt


def gen_prompt(train_df, k=-1):
    if train_df==None:
        return ""
    prompt = "What is the correct answer to this question:\n"
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt

@torch.no_grad()
def eval_baseline(args, subject, model, tokenizer, test_df):
    cors = []
    all_probs = []
    answers = choices[: test_df.shape[1] - 2]

    for i in tqdm(range(test_df.shape[0])):
        # get prompt and make sure it fits
        k = args.ntrain
        
        prompt_end = "What is the correct answer to this question:\n"+format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(None, k)
        prompt = train_prompt + prompt_end
        
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

        # while input_ids.shape[-1] > 2048:
        #     k -= 1
        #     train_prompt = gen_prompt(None, k)
        #     prompt = train_prompt + prompt_end
        #     input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

        label = test_df.iloc[i, 2]

        logits = model(
            input_ids=input_ids,
        ).logits[:,-1].flatten()

        probs = (
            torch.nn.functional.softmax(
                torch.tensor(
                    [
                        logits[tokenizer("A").input_ids[-1]],
                        logits[tokenizer("B").input_ids[-1]],
                        logits[tokenizer("C").input_ids[-1]],
                        logits[tokenizer("D").input_ids[-1]],
                    ]
                ),
                dim=0,
            )
            .detach()
            .cpu()
            .to(torch.float32)
            .numpy()
        )
        pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(probs)]
        label={0: "A", 1: "B", 2: "C", 3: "D"}[label]
        cor = pred == label
        cors.append(cor)
        all_probs.append(probs)

    acc = np.mean(cors)
    cors = np.array(cors)

    all_probs = np.array(all_probs)
    print("Average accuracy {:.3f} - {}".format(acc, subject))

    return cors, acc, all_probs

@torch.no_grad()
def eval_ctg(args, subject, model, tokenizer, test_df):
    cors = []
    all_probs = []
    answers = choices[: test_df.shape[1] - 2]
    
    insert_layer=[18]

    print('insert layer: ',insert_layer)
    
    # print(logits.shape)
    # print(hiddens.shape)
    
    control_pipeline = pipeline(
        "ctg-control", 
        model=model, 
        tokenizer=tokenizer, 
        layers=insert_layer,
        device=device)
    
    split_hiddens= get_split_hs(model,tokenizer,'/home/chh/repos/my_ctg/instructions/gpqa/{}_2steps_llama.json'.format(subject))
    
    for i in tqdm(range(test_df.shape[0])):
        k = args.ntrain
        
        # get prompt and make sure it fits
        prompt_end = "What is the correct answer to this question:\n"+format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(None, k)
        prompt = train_prompt + prompt_end
        # if the result is not good, please try use chat template
        input_ids = tokenizer(prompt_template(tokenizer,prompt), return_tensors="pt").input_ids.cuda()

        # while input_ids.shape[-1] > 2048:
        #     k -= 1
        #     train_prompt = gen_prompt(dev_df, subject, k)
        #     prompt = train_prompt + prompt_end
        #     input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

        label = test_df.iloc[i, 2]
        
        logits = control_pipeline.get_next_token(input_ids, activations=split_hiddens[i].unsqueeze(0),token_pos=-1)

        probs = (
            torch.nn.functional.softmax(
                torch.tensor(
                    [
                        logits[tokenizer("A").input_ids[-1]],
                        logits[tokenizer("B").input_ids[-1]],
                        logits[tokenizer("C").input_ids[-1]],
                        logits[tokenizer("D").input_ids[-1]],
                    ]
                ),
                dim=0,
            )
            .detach()
            .cpu()
            .to(torch.float32)
            .numpy()
        )
        pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(probs)]
        label={0: "A", 1: "B", 2: "C", 3: "D"}[label]
        cor = pred == label
        cors.append(cor)
        all_probs.append(probs)

    acc = np.mean(cors)
    cors = np.array(cors)

    all_probs = np.array(all_probs)
    print("Average accuracy {:.3f} - {}".format(acc, subject))

    return cors, acc, all_probs

def main(args):

    tokenizer = AutoTokenizer.from_pretrained(args.model_path,padding_side='left')
    model = LlamaForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16).to(device)
    tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    model = model.eval()
    
    subjects=['diamond', 'main']

    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if not os.path.exists(os.path.join(args.save_dir, "results_{}".format(args.model))):
        os.makedirs(os.path.join(args.save_dir, "results_{}".format(args.model)))

    all_cors=[]    
    for subject in subjects:
        ds = load_dataset(args.data_dir,subject)['train']
        test_df =ds.to_pandas()
        
        if args.mode=='baseline':
            cors, acc, probs = eval_baseline(args, subject, model, tokenizer, test_df)
        elif args.mode=='ctg':
            cors, acc, probs = eval_ctg(args, subject, model, tokenizer, test_df)
            
        
        all_cors.append(cors)

        test_df["{}_correct".format(args.model)] = cors
        for j in range(probs.shape[1]):
            choice = choices[j]
            test_df["{}_choice{}_probs".format(args.model, choice)] = probs[:, j]
        test_df.to_csv(
            os.path.join(
                args.save_dir, "results_{}".format(args.model), "{}.csv".format(subject)
            ),
            index=None,
        )

    weighted_acc = np.mean(np.concatenate(all_cors))
    print("Average accuracy: {:.3f}".format(weighted_acc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=0)
    parser.add_argument("--data_dir", "-d", type=str, default="/data1/chh/datasets/jeggers/gpqa_formatted")
    parser.add_argument("--save_dir", "-s", type=str, default="results/gpqa/res1")
    parser.add_argument(
        "--model_path",
        "-m",
        type=str,
        default="/data1/chh/models/meta-llama/Meta-Llama-3-8B-Instruct",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Meta-Llama-3-8B-Instruct",
    )
    parser.add_argument("--mode",type=str,default='ctg')
    
    args = parser.parse_args()
    main(args) 