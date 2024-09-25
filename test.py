import json

from datasets import load_dataset

file1='/home/chh/repos/my_ctg/instructions/gsm8k/gsm8k_2steps_gpt.json'
file2='/home/chh/repos/my_ctg/instructions/gsm8k/gsm8k_2steps_llama.json'
file3='/home/chh/repos/my_ctg/instructions/gsm8k/gsm8k_2steps_gpt_new.json'
file4='/home/chh/repos/my_ctg/instructions/gsm8k/gsm8k_2steps_llama_new.json'
ds = load_dataset("/data1/chh/datasets/openai/gsm8k",'main')
ds=ds['test']

with open(file1, "r", encoding="utf-8") as f:
    data1=json.load(f)
with open(file2,'r',encoding='utf-8') as f:
    data2=json.load(f)
    
for i in ds:
    
    for j in data1:
        if i['question'] == j['question']:
            j['answer']=i['answer']
    for k in data2:
        if i['question'] == k['question']:
            k['answer']=i['answer']
with open(file3, "w", encoding="utf-8") as output_file:
    json.dump(data1, output_file, ensure_ascii=False, indent=4)
with open(file4,'w',encoding='utf-8') as output_file:
    json.dump(data2, output_file, ensure_ascii=False, indent=4)