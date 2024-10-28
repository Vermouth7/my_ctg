#!/bin/bash

# 循环控制 insert_layers 从 5 到 25
for i in $(seq 26 32)
do
    INSERT_LAYERS="--insert_layers [20,$i]"
    
    CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes=4 --main_process_port 29501 -m lm_eval --model hf_wrap \
    --model_args pretrained=/data1/chh/models/meta-llama/Meta-Llama-3-8B-Instruct \
    --tasks ifeval \
    --batch_size 1 \
    --num_fewshot 0 \
    --output_path /home/chh/repos/my_ctg/results/ifeval/ \
    --log_samples \
    --my_mode 0 \
    --split_file /home/chh/repos/my_ctg/instructions/ifeval/ifeval_2steps_llama_2.json \
    --apply_chat_template \
    --gen_kwargs top_p=0.90,temperature=0.6 \
    $INSERT_LAYERS \
    --normalize \
    --operator 'linear_comb' \
    --coef 0.5
done

for i in $(seq 1 4)
do
    INSERT_LAYERS="--insert_layers [20,$i]"
    
    CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes=4 --main_process_port 29501 -m lm_eval --model hf_wrap \
    --model_args pretrained=/data1/chh/models/meta-llama/Meta-Llama-3-8B-Instruct \
    --tasks ifeval \
    --batch_size 1 \
    --num_fewshot 0 \
    --output_path /home/chh/repos/my_ctg/results/ifeval/ \
    --log_samples \
    --my_mode 0 \
    --split_file /home/chh/repos/my_ctg/instructions/ifeval/ifeval_2steps_llama_2.json \
    --apply_chat_template \
    --gen_kwargs top_p=0.90,temperature=0.6 \
    $INSERT_LAYERS \
    --normalize \
    --operator 'linear_comb' \
    --coef 0.5
done
