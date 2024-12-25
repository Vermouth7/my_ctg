CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes=4 --main_process_port 29502 -m lm_eval --model hf_wrap \
    --model_args pretrained=/data1/chh/models/meta-llama/Meta-Llama-3-8B-Instruct \
    --tasks gsm8k_cot_llama \
    --batch_size 1 \
    --num_fewshot 0 \
    --output_path /home/chh/repos/my_ctg/results/gsm8k/ \
    --log_samples \
    --my_mode 9 \
    --apply_chat_template \
    --gen_kwargs max_gen_toks=512 \
    --discriminator /data1/chh/my_ctg/classifier/gsm8k/logistic_regression_model6.pkl

