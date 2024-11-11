# task: gsm8k, gsm8k_cot_llama, minerva_math,ifeval
python  main.py   --tasks ds1000-all-completion  --allow_code_execution --n_samples 1  --load_generations_path /home/chh/repos/my_ctg/results/ds1000/_ds1000-all-completion.json  --model /data1/chh/models/meta-llama/Meta-Llama-3-8B-Instruct

CUDA_VISIBLE_DEVICES= accelerate launch --num_processes=4 -m lm_eval --model hf \
    --model_args pretrained=/data1/chh/models/meta-llama/Meta-Llama-3-8B-Instruct \
    --tasks ifeval \
    --batch_size auto \
    --num_fewshot 0 \
    --output_path /home/chh/repos/my_ctg/results/ifeval/ \
    --log_samples \
    --seed 42

CUDA_VISIBLE_DEVICES=0 lm_eval --model vllm \
    --model_args pretrained=/data1/chh/models/meta-llama/Meta-Llama-3-8B-Instruct,dtype=auto,gpu_memory_utilization=0.9 \
    --tasks gsm8k \
    --batch_size auto \
    --output_path /home/chh/repos/my_ctg/results/gsm8k/ \
    --log_samples \
    --seed 42

CUDA_VISIBLE_DEVICES=0 lm_eval --model hf_wrap \
    --model_args pretrained=/data1/chh/models/meta-llama/Llama-3.2-1B \
    --tasks gsm8k_cot_llama \
    --batch_size 1 \
    --num_fewshot 0 \
    --output_path /home/chh/repos/my_ctg/results/gsm8k/ \
    --log_samples \
    --my_mode 0 \
    --split_file /home/chh/repos/my_ctg/instructions/gsm8k/gsm8k_2steps_llama_5.json

CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch --num_processes=3 --main_process_port 29501 -m lm_eval --model hf_wrap \
    --model_args pretrained=/data1/chh/models/meta-llama/Meta-Llama-3-8B-Instruct \
    --tasks gsm8k \
    --batch_size 1 \
    --num_fewshot 0 \
    --output_path /home/chh/repos/my_ctg/results/gsm8k/ \
    --log_samples \
    --my_mode 1 \
    --apply_chat_template \
    --gen_kwargs top_p=0.90,temperature=0.6 \
    --insert_layers [3,20] \
    --normalize \
    --operator 'linear_comb' \
    --coef 0.5 \
    --split_file /home/chh/repos/my_ctg/instructions/gsm8k/gsm8k_2steps_llama_5.json

CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch --num_processes=3 --main_process_port 29501 -m lm_eval --model hf_wrap \
    --model_args pretrained=/data1/chh/models/meta-llama/Meta-Llama-3-8B-Instruct \
    --tasks gsm8k_cot_llama \
    --batch_size 1 \
    --num_fewshot 8 \
    --output_path /home/chh/repos/my_ctg/results/gsm8k/ \
    --log_samples \
    --my_mode 1 \
    --apply_chat_template \
    --gen_kwargs top_p=0.90,temperature=0.6 \
    --insert_layers [3,20] \
    --normalize \
    --operator 'linear_comb' \
    --coef 0.5 \
    --split_file /home/chh/repos/my_ctg/instructions/gsm8k/gsm8k_2steps_llama_5.json \
    --fewshot_as_multiturn

CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch --num_processes=3 --main_process_port 29501 -m lm_eval --model hf_wrap \
    --model_args pretrained=/data1/chh/models/meta-llama/Meta-Llama-3-8B-Instruct \
    --tasks minerva_math \
    --batch_size 1 \
    --num_fewshot 0 \
    --output_path /home/chh/repos/my_ctg/results/math/ \
    --log_samples \
    --my_mode 1 \
    --apply_chat_template \
    --gen_kwargs top_p=0.90,temperature=0.6 \
    --insert_layers [3,20] \
    --normalize \
    --operator 'linear_comb' \
    --coef 0.5 \
    --split_file /home/chh/repos/my_ctg/instructions/math/math_2steps_llama.json \
    --fewshot_as_multiturn


CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch --num_processes=3 --main_process_port 29501 -m lm_eval --model hf_wrap \
    --model_args pretrained=/data1/chh/models/meta-llama/Meta-Llama-3-8B-Instruct \
    --tasks minerva_math \
    --batch_size 1 \
    --num_fewshot 4 \
    --output_path /home/chh/repos/my_ctg/results/math/ \
    --log_samples \
    --my_mode 1 \
    --apply_chat_template \
    --gen_kwargs top_p=0.90,temperature=0.6 \
    --insert_layers [3,20] \
    --normalize \
    --operator 'linear_comb' \
    --coef 0.5 \
    --split_file /home/chh/repos/my_ctg/instructions/math/math_2steps_llama.json

CUDA_VISIBLE_DEVICES=2 opencompass --hf-type chat --hf-path /data1/chh/models/meta-llama/Meta-Llama-3-8B-Instruct --datasets gsm8k_0shot_v2_gen_a58960  -a vllm
CUDA_VISIBLE_DEVICES=3 opencompass --hf-type chat --hf-path /data1/chh/models/meta-llama/Meta-Llama-3-8B-Instruct --datasets math_0shot_gen_393424  -a vllm
CUDA_VISIBLE_DEVICES=4 opencompass --hf-type chat --hf-path /data1/chh/models/meta-llama/Meta-Llama-3-8B-Instruct --datasets math_4shot_example_from_google_research  -a vllm
CUDA_VISIBLE_DEVICES=5 opencompass --hf-type chat --hf-path /data1/chh/models/meta-llama/Meta-Llama-3-8B-Instruct --datasets ds1000_compl_gen_cbc84f  -a vllm
CUDA_VISIBLE_DEVICES=6 opencompass --hf-type chat --hf-path /data1/chh/models/meta-llama/Meta-Llama-3-8B-Instruct --datasets humaneval_gen  -a vllm

CUDA_VISIBLE_DEVICE=0 evalplus.evaluate --model "/data1/chh/models/meta-llama/Meta-Llama-3-8B-Instruct" \
                  --dataset humaneval \
                  --backend vllm \
                   