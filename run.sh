# !/bin/bash

# for i in {11..32}
# do
#     if [ "$i" -eq 3 ]; then
#         continue
#     fi
#     INSERT_LAYERS="--insert_layers [3,$i]"

#     CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch --num_processes=3 --main_process_port 29501 -m lm_eval --model hf_wrap \
#         --model_args pretrained=/data1/chh/models/meta-llama/Meta-Llama-3-8B-Instruct \
#         --tasks ifeval \
#         --batch_size 1 \
#         --num_fewshot 0 \
#         --output_path /home/chh/repos/my_ctg/results/ifeval/ \
#         --log_samples \
#         --my_mode 1 \
#         --apply_chat_template \
#         $INSERT_LAYERS \
#         --normalize \
#         --operator 'replace' \
#         --split_file /home/chh/repos/my_ctg/instructions/ifeval/ifeval_2steps_llama_2.json
# done


# CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --num_processes=4 --main_process_port 29501 -m lm_eval --model hf_wrap \
#     --model_args pretrained=/data1/chh/models/meta-llama/Meta-Llama-3-8B-Instruct \
#     --tasks gsm8k_cot_llama \
#     --batch_size 1 \
#     --num_fewshot 8 \
#     --output_path /home/chh/repos/my_ctg/results/gsm8k/ \
#     --log_samples \
#     --my_mode 1 \
#     --apply_chat_template \
#     --insert_layers [3,20] \
#     --normalize \
#     --operator 'linear_comb' \
#     --coef 0.5 \
#     --split_file /home/chh/repos/my_ctg/instructions/gsm8k/gsm8k_2steps_llama_5.json \
#     --fewshot_as_multiturn \
#     --gen_kwargs max_gen_toks=512

# CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch --num_processes=3 --main_process_port 29501 -m lm_eval --model hf_wrap \
#     --model_args pretrained=/data1/chh/models/meta-llama/Meta-Llama-3-8B-Instruct \
#     --tasks gsm8k_cot_llama \
#     --batch_size 1 \
#     --num_fewshot 8 \
#     --output_path /home/chh/repos/my_ctg/results/gsm8k/ \
#     --log_samples \
#     --my_mode 2 \
#     --apply_chat_template \
#     --insert_layers [3,20] \
#     --normalize \
#     --operator 'linear_comb' \
#     --coef 0.5 \
#     --split_file /home/chh/repos/my_ctg/instructions/gsm8k/gsm8k_2steps_llama_5.json \
#     --fewshot_as_multiturn \
#     --gen_kwargs max_gen_toks=512 \
#     --discriminator /data1/chh/my_ctg/classifier/gsm8k/logistic_regression_model7.pkl

CUDA_VISIBLE_DEVICES=2,3,4,5 accelerate launch --num_processes=4 --main_process_port 29501 -m lm_eval --model hf_wrap \
    --model_args pretrained=/data1/chh/models/meta-llama/Meta-Llama-3-8B-Instruct \
    --tasks ifeval \
    --batch_size 1 \
    --num_fewshot 0 \
    --output_path /home/chh/repos/my_ctg/results/ifeval/ \
    --log_samples \
    --my_mode 1 \
    --apply_chat_template \
    --insert_layers [3] \
    --normalize \
    --operator 'replace' \
    --split_file /home/chh/repos/my_ctg/instructions/ifeval/ifeval_2steps_llama_2.json 


# CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --num_processes=4 --main_process_port 29501 -m lm_eval --model hf_wrap \
#     --model_args pretrained=/data1/chh/models/meta-llama/Meta-Llama-3-8B-Instruct \
#     --tasks ifeval \
#     --batch_size 1 \
#     --num_fewshot 0 \
#     --output_path /home/chh/repos/my_ctg/results/ifeval/ \
#     --log_samples \
#     --my_mode 0 \
#     --patch \
#     --apply_chat_template \
#     --split_file /home/chh/repos/my_ctg/instructions/ifeval/ifeval_2steps_llama_2.json


# CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --num_processes=4 --main_process_port 29501 -m lm_eval --model hf_wrap \
#     --model_args pretrained=/data1/chh/models/meta-llama/Meta-Llama-3-8B-Instruct \
#     --tasks ifeval \
#     --batch_size 1 \
#     --num_fewshot 0 \
#     --output_path /home/chh/repos/my_ctg/results/ifeval/ \
#     --log_samples \
#     --my_mode 1 \
#     --apply_chat_template \
#     --insert_layers [3] \
#     --normalize \
#     --patch \
#     --operator 'replace' \
#     --split_file /home/chh/repos/my_ctg/instructions/ifeval/ifeval_2steps_llama_2.json

# CUDA_VISIBLE_DEVICES=4 accelerate launch --num_processes=1 --main_process_port 29501 -m lm_eval --model hf_wrap \
#     --model_args pretrained=/data1/chh/models/meta-llama/Meta-Llama-3-8B-Instruct \
#     --tasks ifeval \
#     --batch_size 1 \
#     --num_fewshot 0 \
#     --output_path /home/chh/repos/my_ctg/results/ifeval/ \
#     --log_samples \
#     --my_mode 6 \
#     --apply_chat_template \
#     --insert_layers [3] \
#     --normalize \
#     --patch \
#     --operator 'replace' \
#     --split_file /home/chh/repos/my_ctg/instructions/ifeval/ifeval_2steps_llama_2.json \
#     --discriminator /data1/chh/models/model_sft/llama3-8b/cls/checkpoint-231


# CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch --num_processes=3 --main_process_port 29501 -m lm_eval --model hf_wrap \
#     --model_args pretrained=/data1/chh/models/meta-llama/Meta-Llama-3-8B-Instruct \
#     --tasks math_0shot_cot \
#     --batch_size 1 \
#     --output_path /home/chh/repos/my_ctg/results/math/ \
#     --log_samples \
#     --my_mode 1 \
#     --insert_layers [3,20] \
#     --normalize \
#     --operator 'linear_comb' \
#     --coef 0.5 \
#     --gen_kwargs max_gen_toks=512 \
#     --split_file /home/chh/repos/my_ctg/instructions/math/math_2steps_llama.json

# CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch --num_processes=3 --main_process_port 29501 -m lm_eval --model hf_wrap \
#     --model_args pretrained=/data1/chh/models/meta-llama/Meta-Llama-3-8B-Instruct \
#     --tasks math_0shot_cot \
#     --batch_size 2 \
#     --output_path /home/chh/repos/my_ctg/results/math/ \
#     --log_samples \
#     --my_mode 2 \
#     --insert_layers [3,20] \
#     --normalize \
#     --operator 'linear_comb' \
#     --coef 0.5 \
#     --gen_kwargs max_gen_toks=512 \
#     --split_file /home/chh/repos/my_ctg/instructions/math/math_2steps_llama.json \
#     --discriminator /data1/chh/my_ctg/classifier/math/logistic_regression_model.pkl



# CUDA_VISIBLE_DEVICES=1 lm_eval --model vllm \
#     --model_args pretrained=/data1/chh/models/meta-llama/Meta-Llama-3-8B-Instruct,add_bos_token=True,seed=42 \
#     --tasks math_0shot_cot \
#     --batch_size auto \
#     --output_path /home/chh/repos/my_ctg/results/math/ \
#     --log_samples \



# CUDA_VISIBLE_DEVICES=1 lm_eval --model vllm \
#     --model_args pretrained=/data1/chh/models/meta-llama/Meta-Llama-3-8B-Instruct \
#     --tasks gsm8k_cot_llama_train \
#     --batch_size auto \
#     --output_path /home/chh/repos/my_ctg/results/gsm8k_act/ \
#     --log_samples \
#     --apply_chat_template \
#     --num_fewshot 8 \
#     --fewshot_as_multiturn \
#     --gen_kwargs max_gen_toks=512

# CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --num_processes=4 --main_process_port 29501 -m lm_eval --model hf_wrap \
#     --model_args pretrained=/data1/chh/models/meta-llama/Meta-Llama-3-8B-Instruct \
#     --tasks gsm8k_cot_llama \
#     --batch_size 1 \
#     --output_path /home/chh/repos/my_ctg/results/gsm8k/ \
#     --log_samples \
#     --my_mode 1 \
#     --apply_chat_template \
#     --num_fewshot 8 \
#     --fewshot_as_multiturn \
#     --gen_kwargs max_gen_toks=512 \
#     --insert_layers [18] \
#     --normalize \
#     --operator 'replace' \
#     --split_file /home/chh/repos/my_ctg/instructions/gsm8k/gsm8k_2steps_llama_5.json

# CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --num_processes=4 --main_process_port 29501 -m lm_eval --model hf_wrap \
#     --model_args pretrained=/data1/chh/models/meta-llama/Meta-Llama-3-8B-Instruct \
#     --tasks gsm8k_cot_llama \
#     --batch_size 1 \
#     --output_path /home/chh/repos/my_ctg/results/gsm8k/ \
#     --log_samples \
#     --apply_chat_template \
#     --my_mode 2 \
#     --num_fewshot 8 \
#     --fewshot_as_multiturn \
#     --gen_kwargs max_gen_toks=512 \
#     --insert_layers [3] \
#     --normalize \
#     --operator 'replace' \
#     --split_file /home/chh/repos/my_ctg/instructions/gsm8k/gsm8k_2steps_llama_5.json \
#     --discriminator /data1/chh/my_ctg/classifier/gsm8k/logistic_regression_model6.pkl

