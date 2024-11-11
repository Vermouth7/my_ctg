CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes=4 --main_process_port 29502 -m lm_eval --model hf_wrap \
    --model_args pretrained=/data1/chh/models/meta-llama/Meta-Llama-3-8B-Instruct \
    --tasks gsm8k_cot_llama \
    --batch_size 1 \
    --num_fewshot 8 \
    --output_path /home/chh/repos/my_ctg/results/gsm8k/ \
    --log_samples \
    --my_mode 0 \
    --split_file /home/chh/repos/my_ctg/instructions/gsm8k/gsm8k_2steps_llama_5.json \
    --apply_chat_template \
    --gen_kwargs top_p=0.90,temperature=0.6 \
    --insert_layers [3,20] \
    --normalize \
    --operator 'linear_comb' \
    --coef 0.5 \
    --fewshot_as_multiturn \
    --apply_chat_template

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes=4 --main_process_port 29502 -m lm_eval --model hf_wrap \
    --model_args pretrained=/data1/chh/models/meta-llama/Meta-Llama-3-8B-Instruct \
    --tasks gsm8k \
    --batch_size 1 \
    --num_fewshot 0 \
    --output_path /home/chh/repos/my_ctg/results/gsm8k/ \
    --log_samples \
    --my_mode 0 \
    --split_file /home/chh/repos/my_ctg/instructions/gsm8k/gsm8k_2steps_llama_5.json \
    --apply_chat_template \
    --gen_kwargs top_p=0.90,temperature=0.6 \
    --insert_layers [3,20] \
    --normalize \
    --operator 'linear_comb' \
    --coef 0.5 \
    --apply_chat_template

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes=4 --main_process_port 29502 -m lm_eval --model hf_wrap \
    --model_args pretrained=/data1/chh/models/meta-llama/Meta-Llama-3-8B-Instruct \
    --tasks minerva_math \
    --batch_size 1 \
    --num_fewshot 0 \
    --output_path /home/chh/repos/my_ctg/results/math/ \
    --log_samples \
    --my_mode 0 \
    --split_file /home/chh/repos/my_ctg/instructions/math/math_2steps_llama.json \
    --apply_chat_template \
    --gen_kwargs top_p=0.90,temperature=0.6 \
    --insert_layers [3,20] \
    --normalize \
    --operator 'linear_comb' \
    --coef 0.5 \
    --apply_chat_template

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes=4 --main_process_port 29502 -m lm_eval --model hf_wrap \
    --model_args pretrained=/data1/chh/models/meta-llama/Meta-Llama-3-8B-Instruct \
    --tasks minerva_math \
    --batch_size 1 \
    --num_fewshot 4 \
    --output_path /home/chh/repos/my_ctg/results/math/ \
    --log_samples \
    --my_mode 0 \
    --split_file /home/chh/repos/my_ctg/instructions/math/math_2steps_llama.json \
    --apply_chat_template \
    --gen_kwargs top_p=0.90,temperature=0.6 \
    --insert_layers [3,20] \
    --normalize \
    --operator 'linear_comb' \
    --coef 0.5 \
    --fewshot_as_multiturn \
    --apply_chat_template