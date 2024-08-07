export CUDA_VISIBLE_DEVICES=6

model="Meta-Llama-3-8B-Instruct_sft5_380"
model_category="open"  # choices=['api', 'open']
model_path='/data1/chh/models/model_sft/llama3-8b/merge/sft5_380'
task="sentiment"  # choices=["sentiment", "topic", "multi", "length", "keyword", "detoxic"]
mode="generate"  # choices=['generate', 'generate_few_shot']
save_fold='results_zero_shot'

python generate.py --model=$model --model_category=$model_category --task=$task --mode=$mode --model_path=$model_path --save_fold=$save_fold

model="Meta-Llama-3-8B-Instruct_sft5_380"
model_category="open"  # choices=['api', 'open']
model_path='/data1/chh/models/model_sft/llama3-8b/merge/sft5_380'
task="topic"  # choices=["sentiment", "topic", "multi", "length", "keyword", "detoxic"]
mode="generate"  # choices=['generate', 'generate_few_shot']
save_fold='results_zero_shot'

python generate.py --model=$model --model_category=$model_category --task=$task --mode=$mode --model_path=$model_path --save_fold=$save_fold

model="Meta-Llama-3-8B-Instruct_sft5_380"
model_category="open"  # choices=['api', 'open']
model_path='/data1/chh/models/model_sft/llama3-8b/merge/sft5_380'
task="multi"  # choices=["sentiment", "topic", "multi", "length", "keyword", "detoxic"]
mode="generate"  # choices=['generate', 'generate_few_shot']
save_fold='results_zero_shot'

python generate.py --model=$model --model_category=$model_category --task=$task --mode=$mode --model_path=$model_path --save_fold=$save_fold



