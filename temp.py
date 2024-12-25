import json

import pandas as pd

df = pd.read_parquet('/home/chh/repos/lm-evaluation-harness/lm_eval/tasks/llama3_math/dataset/joined_math.parquet')
# with open('/home/chh/repos/my_ctg/instructions/math/math_message_4.json','r') as f:
#     data=json.load(f)
# cnt=0
# for j in data:
#     for i in df['input_question']:
#         if j['problem'] in i:
#             cnt+=1
# print(cnt)
