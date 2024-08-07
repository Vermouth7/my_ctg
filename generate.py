import argparse
import json
import os
# os.environ['CUDA_VISIBLE_DEVICES']='0'
import random

import numpy as np
import torch
from tqdm import tqdm
from transformers import (AutoConfig, AutoModel, AutoModelForCausalLM,
                          AutoModelForSequenceClassification, AutoTokenizer,
                          GenerationConfig, LlamaForCausalLM, LlamaTokenizer)
from vllm import LLM, SamplingParams

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

def prompt_template(tokenizer,message: str, system_prompt: str) -> str:
    messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": message},
    ]
    
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


def get_test_data(args):
    tasks = []
    with open(f'/home/chh/repos/my_ctg/instructions/test/{args.task}_lite.jsonl', 'r') as f:
        for line in f.readlines():
            data = json.loads(line)
            tasks.append(data)
    return tasks


def get_open_model(args):
    tokenizer = None
    model = None
    if 'GLM' in args.model:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True).half().cuda()
    elif 'baichuan' in args.model:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="auto", trust_remote_code=True)
    else:
        model = LLM(model=args.model_path,gpu_memory_utilization=0.90)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    return model, tokenizer


def main(args):
    system_content = "You are performing a test of controlled text generation. Generate text according to the following instruction and generate whatever you want, no other requirements:"
    
    result_dir = f'./{args.save_fold}/{args.model}/{args.task}/'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    result_path = result_dir + f'generation.jsonl'

    if args.mode == 'generate_few_shot':
        # read few shot demos
        system_content = "You are performing a test of controlled text generation. Generate text according to the following 6 instruction and generate whatever you want, no other requirements:"
        with open(f'./data/few_shot/{args.task}.txt', 'r') as f:
            few_shot_examples = f.read()

    instructions = get_test_data(args)
    tokenizer = None

    if args.task == 'detoxic':
        args.max_len = 75

    if args.model_category == 'api':
        print()

    else:
        model, tokenizer = get_open_model(args)
        prompts=[]
        sampling_params = SamplingParams(temperature=0.7, top_p=0.9,max_tokens=300)
        tasks=[]
        # model.to(args.device)
        # model.eval()
        for i in tqdm(range(len(instructions))):
            task = instructions[i]
            prompt = None
            if args.mode == 'generate':
                prompt = f"Instruction: {task['instruction']}\nResponse:"
                prompt = prompt_template(tokenizer,prompt, system_content)
            else:
                prompt = few_shot_examples + f"\n6.Instruction: {task['instruction']}\nResponse:"
                prompt = prompt_template(tokenizer,prompt, system_content)
            tasks.append(task)
            prompts.append(prompt)
        outputs = model.generate(prompts, sampling_params)
        # Print the outputs.
        for output,task in zip(outputs,tasks):
            generated_text = output.outputs[0].text
            result_dict = get_result_dict(args, task, generated_text)
            # save results
            with open(result_path, 'a+') as f:
                f.write(json.dumps(result_dict))
                f.write('\n')


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--model", default='ChatGPT', type=str)
    parser.add_argument("--model_path", default='', type=str)
    parser.add_argument("--model_category", default='api', type=str, choices=['api', 'open'])
    parser.add_argument("--task", default='sentiment', type=str, choices=["sentiment", "topic", "multi", "length", "keyword", "detoxic"])
    parser.add_argument("--mode", default='generate', type=str, choices=['generate', 'generate_few_shot'])
    parser.add_argument("--save_fold", default='results_zero_shot', type=str)
    # decoding
    parser.add_argument("--top_p", default=0.9, type=float)
    parser.add_argument("--max_len", default=300, type=int)
    parser.add_argument("--length_penalty", default=1, type=int)
    parser.add_argument("--repetition_penalty", default=1, type=int)
    parser.add_argument("--temperature", default=0.6, type=float)
    # gpu
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument('--local_rank', default=0, type=int, help='node rank for distributed training')
    parser.add_argument('--gpus', default=1, type=int, help='number of gpus per node')
    parser.add_argument('--nodes', default=1, type=int, metavar='N')
    args = parser.parse_args()
    set_seed(args)

    main(args)
