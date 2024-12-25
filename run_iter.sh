# !/bin/bash
# for i in {1..32}
# do
#     INSERT_LAYERS="--insert_layers [$i]"

#     CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --num_processes=4 --main_process_port 29501 -m lm_eval --model hf_wrap \
#         --model_args pretrained=/data1/chh/models/meta-llama/Meta-Llama-3-8B-Instruct \
#         --tasks ifeval \
#         --batch_size 1 \
#         --num_fewshot 0 \
#         --output_path /home/chh/repos/my_ctg/results/ifeval_iter/ \
#         --log_samples \
#         --my_mode 1 \
#         --apply_chat_template \
#         $INSERT_LAYERS \
#         --normalize \
#         --operator 'linear_comb' \
#         --coef 0.5 \
#         --split_file /home/chh/repos/my_ctg/instructions/ifeval/ifeval_2steps_llama_4.json
# done


CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --num_processes=4 --main_process_port 29501 -m lm_eval --model hf_wrap \
        --model_args pretrained=/data1/chh/models/meta-llama/Meta-Llama-3-8B-Instruct \
        --tasks ifeval \
        --batch_size 1 \
        --num_fewshot 0 \
        --output_path /home/chh/repos/my_ctg/results/ifeval_iter/ \
        --log_samples \
        --my_mode 1 \
        --apply_chat_template \
        --insert_layers '[32]' \
        --normalize \
        --operator 'linear_comb' \
        --coef 0.5 \
        --split_file /home/chh/repos/my_ctg/instructions/ifeval/ifeval_2steps_llama_4.json


CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --num_processes=4 --main_process_port 29501 -m lm_eval --model hf_wrap \
        --model_args pretrained=/data1/chh/models/meta-llama/Meta-Llama-3-8B-Instruct \
        --tasks ifeval \
        --batch_size 1 \
        --num_fewshot 0 \
        --output_path /home/chh/repos/my_ctg/results/ifeval_iter/ \
        --log_samples \
        --my_mode 1 \
        --apply_chat_template \
        --insert_layers '[32]' \
        --normalize \
        --operator 'linear_comb' \
        --coef 0.6 \
        --split_file /home/chh/repos/my_ctg/instructions/ifeval/ifeval_2steps_llama_4.json


CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --num_processes=4 --main_process_port 29501 -m lm_eval --model hf_wrap \
        --model_args pretrained=/data1/chh/models/meta-llama/Meta-Llama-3-8B-Instruct \
        --tasks ifeval \
        --batch_size 1 \
        --num_fewshot 0 \
        --output_path /home/chh/repos/my_ctg/results/ifeval_iter/ \
        --log_samples \
        --my_mode 1 \
        --apply_chat_template \
        --insert_layers '[32]' \
        --normalize \
        --operator 'linear_comb' \
        --coef 0.7 \
        --split_file /home/chh/repos/my_ctg/instructions/ifeval/ifeval_2steps_llama_4.json


CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --num_processes=4 --main_process_port 29501 -m lm_eval --model hf_wrap \
        --model_args pretrained=/data1/chh/models/meta-llama/Meta-Llama-3-8B-Instruct \
        --tasks ifeval \
        --batch_size 1 \
        --num_fewshot 0 \
        --output_path /home/chh/repos/my_ctg/results/ifeval_iter/ \
        --log_samples \
        --my_mode 1 \
        --apply_chat_template \
        --insert_layers '[32]' \
        --normalize \
        --operator 'linear_comb' \
        --coef 0.8 \
        --split_file /home/chh/repos/my_ctg/instructions/ifeval/ifeval_2steps_llama_4.json

CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --num_processes=4 --main_process_port 29501 -m lm_eval --model hf_wrap \
        --model_args pretrained=/data1/chh/models/meta-llama/Meta-Llama-3-8B-Instruct \
        --tasks ifeval \
        --batch_size 1 \
        --num_fewshot 0 \
        --output_path /home/chh/repos/my_ctg/results/ifeval_iter/ \
        --log_samples \
        --my_mode 1 \
        --apply_chat_template \
        --insert_layers '[32]' \
        --normalize \
        --operator 'linear_comb' \
        --coef 0.9 \
        --split_file /home/chh/repos/my_ctg/instructions/ifeval/ifeval_2steps_llama_4.json


CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --num_processes=4 --main_process_port 29501 -m lm_eval --model hf_wrap \
        --model_args pretrained=/data1/chh/models/meta-llama/Meta-Llama-3-8B-Instruct \
        --tasks ifeval \
        --batch_size 1 \
        --num_fewshot 0 \
        --output_path /home/chh/repos/my_ctg/results/ifeval_iter/ \
        --log_samples \
        --my_mode 1 \
        --apply_chat_template \
        --insert_layers '[32]' \
        --normalize \
        --operator 'linear_comb' \
        --coef 1.0 \
        --split_file /home/chh/repos/my_ctg/instructions/ifeval/ifeval_2steps_llama_4.json


CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --num_processes=4 --main_process_port 29501 -m lm_eval --model hf_wrap \
        --model_args pretrained=/data1/chh/models/meta-llama/Meta-Llama-3-8B-Instruct \
        --tasks ifeval \
        --batch_size 1 \
        --num_fewshot 0 \
        --output_path /home/chh/repos/my_ctg/results/ifeval_iter/ \
        --log_samples \
        --my_mode 1 \
        --apply_chat_template \
        --insert_layers '[32]' \
        --normalize \
        --operator 'linear_comb' \
        --coef 1.1 \
        --split_file /home/chh/repos/my_ctg/instructions/ifeval/ifeval_2steps_llama_4.json


CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --num_processes=4 --main_process_port 29501 -m lm_eval --model hf_wrap \
        --model_args pretrained=/data1/chh/models/meta-llama/Meta-Llama-3-8B-Instruct \
        --tasks ifeval \
        --batch_size 1 \
        --num_fewshot 0 \
        --output_path /home/chh/repos/my_ctg/results/ifeval_iter/ \
        --log_samples \
        --my_mode 1 \
        --apply_chat_template \
        --insert_layers '[32]' \
        --normalize \
        --operator 'linear_comb' \
        --coef 1.2 \
        --split_file /home/chh/repos/my_ctg/instructions/ifeval/ifeval_2steps_llama_4.json
    

CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --num_processes=4 --main_process_port 29501 -m lm_eval --model hf_wrap \
        --model_args pretrained=/data1/chh/models/meta-llama/Meta-Llama-3-8B-Instruct \
        --tasks ifeval \
        --batch_size 1 \
        --num_fewshot 0 \
        --output_path /home/chh/repos/my_ctg/results/ifeval_iter/ \
        --log_samples \
        --my_mode 1 \
        --apply_chat_template \
        --insert_layers '[32]' \
        --normalize \
        --operator 'linear_comb' \
        --coef 1.3 \
        --split_file /home/chh/repos/my_ctg/instructions/ifeval/ifeval_2steps_llama_4.json


CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --num_processes=4 --main_process_port 29501 -m lm_eval --model hf_wrap \
        --model_args pretrained=/data1/chh/models/meta-llama/Meta-Llama-3-8B-Instruct \
        --tasks ifeval \
        --batch_size 1 \
        --num_fewshot 0 \
        --output_path /home/chh/repos/my_ctg/results/ifeval_iter/ \
        --log_samples \
        --my_mode 1 \
        --apply_chat_template \
        --insert_layers '[32]' \
        --normalize \
        --operator 'linear_comb' \
        --coef 1.4 \
        --split_file /home/chh/repos/my_ctg/instructions/ifeval/ifeval_2steps_llama_4.json


CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --num_processes=4 --main_process_port 29501 -m lm_eval --model hf_wrap \
        --model_args pretrained=/data1/chh/models/meta-llama/Meta-Llama-3-8B-Instruct \
        --tasks ifeval \
        --batch_size 1 \
        --num_fewshot 0 \
        --output_path /home/chh/repos/my_ctg/results/ifeval_iter/ \
        --log_samples \
        --my_mode 1 \
        --apply_chat_template \
        --insert_layers '[32]' \
        --normalize \
        --operator 'linear_comb' \
        --coef 1.5 \
        --split_file /home/chh/repos/my_ctg/instructions/ifeval/ifeval_2steps_llama_4.json


CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --num_processes=4 --main_process_port 29501 -m lm_eval --model hf_wrap \
        --model_args pretrained=/data1/chh/models/meta-llama/Meta-Llama-3-8B-Instruct \
        --tasks ifeval \
        --batch_size 1 \
        --num_fewshot 0 \
        --output_path /home/chh/repos/my_ctg/results/ifeval_iter/ \
        --log_samples \
        --my_mode 1 \
        --apply_chat_template \
        --insert_layers '[32]' \
        --normalize \
        --operator 'linear_comb' \
        --coef 2.0 \
        --split_file /home/chh/repos/my_ctg/instructions/ifeval/ifeval_2steps_llama_4.json