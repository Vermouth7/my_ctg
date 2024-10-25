import argparse
import json
import os
import random
import re
import sys
import time
from collections import defaultdict

import numpy as np
import openai
import pandas as pd
import torch
from rouge import Rouge
from utils_dataset import query_qwen_dataset

topics = [
    'arts_&_culture', 'business_&_entrepreneurs', 'celebrity_&_pop_culture', 'diaries_&_daily_life', 'family', 
    'fashion_&_style', 'film_tv_&_video', 'fitness_&_health', 'food_&_dining', 'gaming',
    'learning_&_educational', 'music', 'news_&_social_concern', 'relationships',
    'science_&_technology', 'sports', 'travel_&_adventure', 'youth_&_student_life'
]
sentiments = ['anger', 'disgust', 'fear', 'joy', 'negative', 'neutral', 'positive', 'sadness', 'surprise']
sent_style = [
    'Formal', 'informal', 'concise', 'verbose', 'polite', 'causal', 'word conversion'
]
sent_style_extend = [
    "Flowery","Ornate","Poetic","Sparse","Rambling","Jargon-filled","Technical","Pithy","Witty",
    "Sarcastic","Humorous","Dramatic","Melodramatic","Colloquial","Idiomatic","Figurative", 
    "Metaphorical","Symbolic","Rhetorical","Persuasive","Eloquent","Laconic","Elevated","Simplistic","Sententious"
]


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


def get_tasks(args):
    tasks = []
    with open(args.instruction_path, 'r') as f:
        for line in f.readlines():
            tasks.append(json.loads(line))
    return tasks


def get_topic_dict():
    topic_dict = {}
    for topic in topics:
        topic_list = topic.split('_&_')
        topic_list = [t.replace('_', ' ') for t in topic_list]
        if len(topic_list) > 1:
            topic_all = ' or '.join(topic_list)
            topic_list.append(topic_all)
        topic_dict[topic] = topic_list
    return topic_dict

def self_instruct(tasks_pool, prompt, pre_nums):
    '''
    expand instructions from seed tasks
    '''
    seed_task_num = len(tasks_pool)
    for idx in range(200):
        if idx < 1:
            instruction_selected = random.sample(tasks_pool, pre_nums)
        else:
            instruction_selected = random.sample(tasks_pool[:seed_task_num], pre_nums)

        random.shuffle(instruction_selected)

        pre_instructions = ''
        i = 1
        for instruct in instruction_selected:
            pre_instructions += f'{i}' + ':[' + instruct + ']\n'
            i += 1
        prompt_to_query = prompt + pre_instructions
        prompt_to_query += f'{i}:'
        # print(prompt_to_query)

        n_completions = 1
        response = query_qwen_dataset(prompt_to_query)

        # extract new instructions
        for j in range(n_completions):
            res = response
            print(res)
            pattern = r"\[(.+?)\]"
            matches = re.findall(pattern, res)
            
            # fail to generate new instructions
            if len(matches) == 0:
                continue
            # save instructions
            with open(args.source_path, 'a+') as f:
                for txt in matches:
                    # filter
                    rouge = Rouge()
                    scores = []
                    for i in tasks_pool:
                        scores.append(rouge.get_scores(txt, i, avg=True)['rouge-l']['f'])
                    if len(scores) > 0 and max(scores) > 0.95:
                        continue
                    if 'list' not in txt:
                        tasks_pool.append(txt)
                    
                    sentiment = random.choice(sentiments)
                    topic_selected = random.choice(topics)
                    topic = random.choice(topic_dict[topic_selected])
                    # data = {'instruction': txt.format(sentiment=sentiment, topic=topic)}
                    if ('{topic}' not in txt) and ('{sentiment}' not in txt):
                        continue
                    data = {'instruction': txt}
                    f.write(json.dumps(data))
                    f.write('\n')
                    print(f'-----extend {len(tasks_pool)} instructions-----')

        if len(tasks_pool) >= 200:
            break

    return tasks_pool


def diversify(tasks_pool, prompt, pre_nums):
    '''
    Diversify seed tasks, use iCL
    '''
    diversified_instructions = []
    topic_dict = get_topic_dict()
    rouge = Rouge()
    combination_counts = defaultdict(int)
    all_combinations = [(topic, sentiment) for topic in topics for sentiment in sentiments]
    random.shuffle(all_combinations)
    
    for topic, sentiment in all_combinations:
        topic_selected=topic
        topic_to_query = random.choice(topic_dict[topic])
        
        while combination_counts[(topic, sentiment)] < 20:
            prompt_to_query = ''
            # choose a topic
            topic_to_query = random.choice(topic_dict[topic])
            sentiment_to_query = sentiment
            
            
            # Due to the large number of categories, in order to cover as many categories as possible with the instructions, 
            # the first 900 non-regenerated instructions(plus seed_tasks is 1000) are not selected, and selection begins after the first 900.
            
            
            example = random.sample(tasks_pool, 1)[0]
            example = example.format(topic=topic_to_query, sentiment=sentiment_to_query)
            
                    
            style = None
            method = random.random()
            if method < 0.2:
                style = random.choice(sent_style_extend)
            elif method < 0.7:
                style = random.choice(sent_style)

            if style is not None:
                if style == 'word conversion':
                    style_query = ', using part-of-speech conversion of words expressing topic or sentiment.'
                else:
                    style_query = f', be {style}'
            else:
                style_query = ''
            prompt_to_query = prompt.format(instruction=example, style=style_query)
            # print(prompt_to_query)
            n_completions = 1

            response = query_qwen_dataset(prompt_to_query)

            # extract new instructions
            for j in range(n_completions):
                res = response
                # print(res)

                pattern = r"\[(.+?)\]"
                match = re.search(pattern, res)
                # fail to generate new instructions
                if match:
                    res = match.group(1)
                else:
                    continue

                # filter undesired instructions
                if 'list' in res:
                    continue

                # filter
                rouge = Rouge()
                scores = []
                for i in diversified_instructions:
                    if type(i) == str:
                        scores.append(rouge.get_scores(res, i, avg=True)['rouge-l']['f'])
                    else:
                        scores.append(rouge.get_scores(res, i['instruction'], avg=True)['rouge-l']['f'])
                if len(scores) > 0 and max(scores) > 0.95:
                    continue

                data = {'label1': topic_selected, 'label2': sentiment_to_query, 'instruction': res}
                diversified_instructions.append(data)
                combination_counts[(topic, sentiment)] += 1
                # save instructions
                with open(args.target_path, 'a+') as f:
                    f.write(json.dumps(data))
                    f.write('\n')

                print(f'-----has extend {len(diversified_instructions)} instructions-----')
        

    return diversified_instructions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--seed_tasks", default='/home/chh/repos/CoDI-Eval/data/seed_task/multi/multi.json', type=str)
    parser.add_argument("--source_path", default='./train/multi/multi.jsonl', type=str)
    parser.add_argument("--target_path", default='./train/multi/multi_diversified.jsonl', type=str)
    parser.add_argument("--mode", default='diversify', type=str)

    args = parser.parse_args()
    set_seed(args)

    if args.mode == 'self-instruct':
        prompt = "Please respond in plain text in English. In the instructions of multi-aspect controlled text generation, you need to include a sentiment and a topic. " \
        "The generated text should be generally represented by generalized words such as text, something, etc. " \
        "Do not use words that indicate text categories such as story, poem, summary. Do not use words that limit the length of text such as article, passage, short paragraphs. " \
        "The sentiment is represented by {sentiment} in the instructions of the Multi-aspect controlled text generation task. " \
        "The topic is represented by {topic} in the instructions of the Multi-aspect controlled text generation task. " \
        "Note that the topic here is a generalized topic, such as science, news, sports, etc. " \
        "Note that {topic} is a noun, {sentiment} is a noun too." \
        "\n\nPlease comp up with 10 diverse Multi-aspect controlled text generation instructions, be concise:\n"

        topic_dict = get_topic_dict()
        seed_tasks = json.load(open(args.seed_tasks, 'r'))
        # save instructions
        with open(args.source_path, 'a+') as f:
            for txt in seed_tasks:
                sentiment = random.choice(sentiments)
                topic_selected = random.choice(topics)
                topic = random.choice(topic_dict[topic_selected])
                data = {'instruction': txt.format(sentiment=sentiment, topic=topic)}
                f.write(json.dumps(data))
                f.write('\n')
        instructions = self_instruct(seed_tasks, prompt, 8)

    elif args.mode == 'diversify':
        seed_tasks = []
        with open(args.source_path, 'r') as f:
            for line in f.readlines():
                data = json.loads(line)
                seed_tasks.append(data['instruction'])

        pre_nums=3
        prompt = "In the instructions of multi-aspect controlled text generation, you need to include a sentiment and a topic. " \
        "The generated text should be generally represented by generalized words such as text, something, etc." \
        "Do not use words that indicate text categories such as story, poem, summary. Do not use words that limit the length of text such as article, passage, short paragraphs." \
        "Note that you cannot generate instructions about {{topic}} itself, such as 'discussing the importance of {{topic}}'. " \
        "Note that the topic here is a generalized topic, such as science, news, sports, etc. " \
        "Note that the rewritten text is placed in [], for example, A:[]." \
        "Please respond in plain text in English. " \
        "\n\nPlease rewrite the Multi-aspect controlled text generation instructions as required:\n" \
        "Q:'I need a piece of writing that captures sadness and pertains to the general category of business. Can you create that?' Rewrite this instruction, be concise\nA:[Create a business-centered text with sadness:]\n" \
        "Q:'Can you share text related to science that conveys positive sentiment with me?' Rewrite this instruction, using part-of-speech conversion of words expressing topic or sentiment.\nA:[Could you please provide me a positive text with a scientific topic?]\n" \
        "Q:'Please give me text that has a topic of sports and a sentiment of joy:' Rewrite this instruction\nA:[I need a joyful text that focuses on sports:]\n" \
        "Q:'{instruction}' Rewrite this instruction in {style} without changing the topic and sentiment features\nA:"
        instructions = diversify(seed_tasks, prompt, pre_nums)

