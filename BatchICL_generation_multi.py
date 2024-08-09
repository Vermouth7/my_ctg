import argparse
import json
import os

os.environ["CUDA_VISIBLE_DEVICES"]='4'
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
                   extract_text, extract_words, get_lemma, get_test_data,
                   load_eval_models, set_seed)

alpha = 0

device = torch.device("cuda")

def format_test_example(df, idx):
    prompt = "Instruction: "+df[idx]['instruction']
    prompt += "\nResponse: "
    return prompt

def format_example(df, idx, include_answer=True):
    prompt = "Instruction:"+df.iloc[idx, 0]
    prompt += "\nResponse:"
    if include_answer:
        prompt += "{}\n\n".format(df.iloc[idx, 1])
    return prompt

    
def gen_prompt(train_df, k=5):
    prompt=''
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt

def compute_metric(output_filename,ls,task,records,eval_models):
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
    for layer in ls:
        acc = 0
        pred_answers = run_results[layer]['pred_answers']
        
        for pred,record in zip(pred_answers,records):
            if task == 'topic':
                score=classify_topic(device=device,model=eval_models[0],tokenizer=eval_models[1],text=pred,label=record['label'])
            elif task=='sentiment':
                score=classify_sentiment(device=device, model1=eval_models[0], model2=eval_models[2], tokenizer1=eval_models[1], tokenizer2=eval_models[3],text=pred , label=record['label'])
            elif task == 'multi':
                score1=classify_topic(device=device,model=eval_models[0],tokenizer=eval_models[1],text=pred,label=record['label1'])
                score2=classify_sentiment(device=device, model1=eval_models[2], model2=eval_models[4], tokenizer1=eval_models[3], tokenizer2=eval_models[5],text=pred , label=record['label2'])
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
    print("ACC-MAX: %.4f" % (max_acc))
    
    return max_acc,best_layer

def prompt_template(tokenizer,message: str) -> str:
    messages = [
    {"role": "system", "content": 'You are performing a test of controlled text generation. Generate text according to the following instruction and generate whatever you want, no other requirements.'},
    {"role": "user", "content": message},
    ]
    
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

def prepare_input(tokenizer, prompts):
    input_tokens = tokenizer(prompts, return_tensors="pt",padding=True)
    input_tokens = {k:input_tokens[k] for k in input_tokens if k in ["input_ids", "attention_mask"]}
    for t in input_tokens:
        if torch.is_tensor(input_tokens[t]):
            input_tokens[t] = input_tokens[t].to(device)
    return input_tokens

def load(ckpt_dir):
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir,padding_side='left')
    model = LlamaForCausalBatchICLLM.from_pretrained(ckpt_dir, torch_dtype=torch.bfloat16).to(device)

    tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    model = model.eval()

    return model, tokenizer
    


def batch_split(prompts, batch_num):
    batch_prompts = []
    mini_batch = []
    for prompt in prompts:
        mini_batch.append(prompt)
        if len(mini_batch) == batch_num:
            batch_prompts.append(mini_batch)
            mini_batch = []
    if len(mini_batch) != 0:
        batch_prompts.append(mini_batch)
    return batch_prompts

def batch_get_hidden(model, tokenizer, prompts,k,n,pos):
    batch_size = k
    answers = []
    tim = 0
    ptt = 0
    logs = 0
    logit = 0
    for batch_input in tqdm(batch_split(prompts, batch_size)):
        encode_inputs = prepare_input(tokenizer, batch_input)
        if ptt == 0:
            ptt = 1
            ins = encode_inputs["input_ids"]
            print(tokenizer.convert_ids_to_tokens(ins[0]))
        with torch.no_grad():
            outp = model(**encode_inputs, output_hidden_states = True, return_dict = True,out_attn=True,idea=0)#["hidden_states"]
            out_logits = outp["logits"][:,-1,:]
            # outp = outp["hidden_before_mlp"]
            outp = outp["hidden_states"]
        #print(outp[0].shape)
        l_s = outp[0][:,pos:,:].unsqueeze(0)
        for i in range(len(outp)):
            out_s = outp[i][:,pos:,:].unsqueeze(0)
            if i==0:
                continue
            l_s = torch.cat((l_s,out_s),dim=0)
        #print("ls:"+str(l_s.shape))
        l_s = torch.transpose(l_s,0,1)
        hidden_j = l_s[0]
        log_j = out_logits[0]
        for j in range(k):
            if j==0:
                continue
            hidden_j = torch.add(hidden_j,l_s[j])
            log_j = torch.add(log_j,out_logits[j])
        hidden_j = hidden_j/k
        log_j = log_j/k
        hidden_j = hidden_j.unsqueeze(0)
        log_j = log_j.unsqueeze(0)
        if tim == 0:
            tim += 1
            hiddens = hidden_j
            logit = log_j
        else:
            hiddens = torch.cat((hiddens,hidden_j),dim=0)
            logit = torch.cat((logit,log_j),dim=0)
    print(hiddens.shape)
    return hiddens,logit
        
def gen_with_h(model, tokenizer, batch_input,insert_layer,insert_positions,hiddens,max_l,insert_le=1):
    ans = []
    pre_l = 0
    encode_inputs = prepare_input(tokenizer, batch_input)
    ins = encode_inputs["input_ids"].to(device)
    bs = ins.shape[0]
    att = encode_inputs["attention_mask"].to(device)
    ct_a = torch.Tensor([1]*bs).to(device)
    ct_a = ct_a.unsqueeze(1)

    ins_p = torch.LongTensor([insert_positions]*bs).to(device)
    ins_le = torch.LongTensor([insert_le]*bs).to(device)
    for _ in range(max_l):
        if pre_l == 0:
            pre_l = ins.shape[1]
        

        # insert_positions -= 1

        with torch.no_grad():
            outp = model(input_ids = ins,attention_mask=att,ins_attn=False, insert_layer=insert_layer,insert_pos=ins_p,insert_hiddens = hiddens,insert_len = ins_le,idea=1,alpha = alpha,use_cache=True)["logits"]
        # outs = outp[:,-1,:]
        # score_ = outs.clone().detach().cpu().numpy()
        # pos_ = torch.LongTensor([np.argmax(score_[i]) for i in range(bs)]).to(device)
        # pos_ = pos_.unsqueeze(1)
        # ins = torch.cat((ins,pos_),dim=1)
        # att = torch.cat((att,ct_a),dim=1)
        # # print(pos_[-1].item())
        # if pos_[-1].item()==2:
        #     break
        #print(as_p)

        outs = outp[:, -1, :]
        pos_ = torch.argmax(outs, dim=-1,keepdim=True)
        ins = torch.cat((ins, pos_), dim=1)
        att = torch.cat((att, ct_a), dim=1)

        
        if (pos_.squeeze(-1) == tokenizer.eos_token_id).all():
            break
        insert_positions -= 1
        ins_p.fill_(insert_positions)
    ps = tokenizer.batch_decode(ins[:,pre_l:], skip_special_tokens=True)

    for x in ps:
        ans.append(x)
    #     if "assistant\n\n" in x:
    #         start_index = x.index("assistant") + len("assistant")
    #         text = x[start_index:].strip()
    #         ans.append(text)
    #     else:
    #         ans.append(x)
        
    return ans
        

def batch_infer_with_hiddens(model, tokenizer, prompts,insert_layer,insert_positions,hiddens,gen_len,insert_le=1):
    batch_size = 8
    answers = []
    u = 0
    for batch_input in tqdm(batch_split(prompts, batch_size)):
        encode_inputs = prepare_input(tokenizer, batch_input)
        ins = encode_inputs["input_ids"]
        bs = ins.shape[0]
        ins_l = torch.LongTensor([insert_layer]*bs).to(device)
        ins_p = torch.LongTensor([insert_positions]*bs).to(device)
        ins_hiddens = hiddens[u:u+bs].to(device)
        u += bs
        ins = encode_inputs["input_ids"]
        o = gen_with_h(model,tokenizer,batch_input,ins_l,insert_positions,ins_hiddens,max_l=gen_len,insert_le=insert_le)
        #print(pos_)
        answers.extend(o)
    answers = [answer for answer in answers]
    return answers



def batch_infer_num(model, tokenizer, prompts):
    batch_size = 8
    answers = []
    tl = 0
    ans_r = []
    u = 0
    for batch_input in tqdm(batch_split(prompts, batch_size)):
        encode_inputs = prepare_input(tokenizer, batch_input)
        ins = encode_inputs["input_ids"]
        if tl==0:
            print(tokenizer.convert_ids_to_tokens(ins[0]))
            tl+=1
        outputs = model.generate(**encode_inputs, do_sample=True,max_new_tokens=300, pad_token_id=tokenizer.pad_token_id)
        p = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for x in p:
            if "assistant\n\n" in x:
                start_index = x.index("assistant") + len("assistant")
                text = x[start_index:].strip()
                ans_r.append(text)
            else:
                ans_r.append(x)

    return ans_r




def get_val_data(task:str):
    tasks = []
    with open(f'/home/chh/repos/CoDI-Eval/data/instructions/val/{task}_val.jsonl', 'r') as f:
        for line in f.readlines():
            data = json.loads(line)
            tasks.append(data)
    return tasks

def get_layer(tokenizer,model,task_name,eval_models):
    gen_len=150
    records = []
    run_results = {}
    ls = []
    lab_pos = -1
    output_filename = 'batchicl_results/run_results_layer_{}_test2.json'.format(task_name)
    val_df=get_val_data(task_name)
    tot_layer = model.config.num_hidden_layers
    conditions=[]
    num_condition=0
    
    for i in range(len(val_df)):
        test_message = format_test_example(val_df, i)
        prompt_test=prompt_template(tokenizer,test_message)
        temp={'prompt':prompt_test}
        
        for key, value in val_df[i].items():
            if key != 'instruction':
                temp[key] = value
        records.append(temp)
        
        if i==0:
            num_condition = sum(1 for key in temp.keys() if 'label' in key)
            
        for k,v in temp.items():
            if 'label' in k:
                prompt_condition="Instruction: Generate a text that fits the condition: {label}. \nResponse:".format(label=v.replace('_', ' ').replace('&', 'and'))
                prompt_condition=prompt_template(tokenizer,prompt_condition)                    
                conditions.append({'prompt':prompt_condition})
        
    for layer in range(0,tot_layer+1):
        ls.append(str(layer))
        print("Now_ly:"+str(layer))
        if layer == 0:
            hiddens,_ = batch_get_hidden(model,tokenizer,[record['prompt'] for record in conditions],num_condition,len(val_df),pos=-1)
            hiddens_ = hiddens
        else:
            hiddens = hiddens_
        if layer == tot_layer:
            ins_ly = -1
        else:
            ins_ly = layer
        
        pred_answers = batch_infer_with_hiddens(model, tokenizer, [record['prompt'] for record in records],ins_ly,-1,hiddens,gen_len=gen_len,insert_le=1)
        run_results[str(layer)] = {'pred_answers':pred_answers}
        
    with open(output_filename, 'w') as f:
        json.dump(run_results, f, ensure_ascii=False, indent=2)
    
    score,best_layer = compute_metric(output_filename,ls,task_name,records,eval_models)
    return best_layer


def compute_baseline(model,tokenizer,task,test_df,eval_models):
    run_results={}
    records=[]
    
    output_filename='batchicl_results/baseline_{}.json'.format(task)
    for i in range(len(test_df)):
        prompt_end = format_test_example(test_df, i)
        prompt = prompt_template(tokenizer,prompt_end)
        temp={'prompt':prompt}
        
        for key, value in test_df[i].items():
            if key != 'instruction':
                temp[key] = value
        records.append(temp)
    
    pred_answers = batch_infer_num(model, tokenizer, [record['prompt'] for record in records])
    run_results['0'] = {'pred_answers':pred_answers}
    
    with open(output_filename, 'w') as f:
        json.dump(run_results, f, ensure_ascii=False, indent=2)
    acc_u, layer= compute_metric(output_filename,['0'],task,records,eval_models)
    
    print('baseline-acc:',acc_u)
    return acc_u

def main(ckpt_dir: str,task_name:str):
    gen_len=150
    run_results = {}
    records = []
    num_condition=0
    model, tokenizer = load(ckpt_dir)
    eval_models=load_eval_models(task_name)
    output_filename = './batchicl_results/run_results_{}_test2.json'.format(task_name)

    start_time = time.time()
    acc = []
    
    test_df=get_test_data(task_name)[:]
    tot_layer = model.config.num_hidden_layers
    
    # compute_baseline(model,tokenizer,task_name,test_df,eval_models)
    # best_layer = get_layer(tokenizer=tokenizer,model=model,task_name=task_name,eval_models=eval_models)
    best_layer=12
    print('best_layer: ',best_layer)
    ls = []
    for layer in range(0,tot_layer+1):
        if layer==0:
            records = []
            conditions = []
            for i in range(len(test_df)):
                test_message = format_test_example(test_df, i)
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
                        prompt_condition="Instruction: Generate a text that fits the condition: {label}. \nResponse:".format(label=v.replace('_', ' ').replace('&', 'and'))
                        prompt_condition=prompt_template(tokenizer,prompt_condition)                    
                        conditions.append({'prompt':prompt_condition})
            
        if layer == 0:
            run_results = {}
            hiddens,logits = batch_get_hidden(model,tokenizer,[condition['prompt'] for condition in conditions],num_condition,len(test_df),pos=-1)
            hiddens_ = hiddens
        else:
            hiddens = hiddens_
        if layer == tot_layer:
            ins_layer = -1
        else:
            ins_layer = layer
        
        if (layer!=int(best_layer)):
            continue
        
        print("Now_ly:"+str(layer))
        print("INs_Layer:"+str(layer))
        ls.append(str(layer))
        pred_answers = batch_infer_with_hiddens(model, tokenizer, [record['prompt'] for record in records],ins_layer,-1,hiddens,gen_len=gen_len,insert_le=1)
        print(pred_answers[:50])
        run_results[str(layer)] = {'pred_answers':pred_answers}
    with open(output_filename, 'w') as f:
        json.dump(run_results, f, ensure_ascii=False, indent=2)

    score, layer= compute_metric(output_filename,ls,task_name,records,eval_models)
    acc.append(score)

    end_time = time.time()
    print("total run time %.2f" % (end_time - start_time))
    acc = np.array(acc)
    acc_mean = np.mean(acc)
    acc_var = np.var(acc)
    print("BatchICL_Mean_Acc:"+str(acc_mean))
    print("BatchICL_Var_Acc:"+str(acc_var))

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', type=str, default='/data1/chh/models/meta-llama/Meta-Llama-3-8B-Instruct')
    # parser.add_argument('--ckpt_dir', type=str, default='/data1/chh/models/model_sft/llama3_lora_sft')
    
    parser.add_argument('--task', type=str, default='multi')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    set_seed(args)
    main(args.ckpt_dir,args.task)