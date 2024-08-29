import json
import os
from collections import Counter

os.environ['CUDA_VISIBLE_DEVICES']= '5'
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = []
        for line in f:
            try:
                item = json.loads(line)
                if isinstance(item, list) and len(item) == 2 and isinstance(item[1], str):
                    data.append(item[1])
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
    return data

def calculate_perplexity(text, model, tokenizer):
    encodings = tokenizer(text, return_tensors='pt')
    max_length = model.config.n_positions
    stride = 512

    lls = []
    for i in range(0, encodings.input_ids.size(1), stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.input_ids.size(1))
        trg_len = end_loc - i
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(model.device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            log_likelihood = outputs.loss * trg_len

        lls.append(log_likelihood)

    perplexity = torch.exp(torch.stack(lls).sum() / end_loc)
    return perplexity.item()

def calculate_distinctness(text, n):
    tokens = text.split()
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    ngram_count = Counter(ngrams)
    distinct_n = len(ngram_count) / len(ngrams) if ngrams else 0
    return distinct_n

def calculate_sentence_level_distinctness(sentences, n):
    total_distinct = 0
    num_sentences = len(sentences)
    
    for sentence in sentences:
        tokens = sentence.split()
        ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
        ngram_count = Counter(ngrams)
        distinct_n = len(ngram_count) / len(ngrams) if ngrams else 0
        total_distinct += distinct_n
    
    return total_distinct / num_sentences if num_sentences > 0 else 0

def main():
    # Load the pre-trained GPT-2 model and tokenizer
    model_name = '/data1/chh/models/openai-community/gpt2-large'
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    
    model.eval()
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    # Load input texts from JSON file
    # input_file = './outputs/generate_single_6.json'  # Replace with your input file path
    input_filename='/home/chh/repos/my_ctg/results/batch_ctg/hs_test7.json'
    with open(input_filename, 'r') as f:
        run_results = json.load(f)
        
    for layer,pred_answers in run_results.items():
        total_perplexity=0
        
        for pred in pred_answers:
            text=pred['text']
            perplexity = calculate_perplexity(text, model, tokenizer)
            # print(perplexity)
            total_perplexity += perplexity
        
        avg_perplexity= total_perplexity / len(pred_answers)
        print("PPL-%2s: %.4f" % (layer,avg_perplexity))
    
if __name__ == "__main__":
    main()
