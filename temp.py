import json
import logging
import os
import random
import re

random.seed(42)

data=[]

with open('/home/chh/repos/my_ctg/results/gsm8k/__data1__chh__models__meta-llama__Meta-Llama-3-8B-Instruct/samples_gsm8k_cot_llama_2024-11-18T15-37-55.772366.jsonl', 'r', encoding='utf-8') as output_file:
    for line in output_file:
        data.append(json.loads(line))
new_data=[]
seee=set()
for i in data:
    if i['exact_match'] == 0.0 and i['doc_id'] not in seee:
        seee.add(i['doc_id'])
        new_data.append(i['doc'])
# test_partition = {
#     "test": new_data
# }
with open('/home/chh/repos/my_ctg/instructions/gsm8k/upperbound.json','w',encoding='utf-8') as f:
    json.dump(new_data, f, ensure_ascii=False, indent=4)
    