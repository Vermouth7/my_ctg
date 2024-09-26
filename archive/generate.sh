# export CUDA_VISIBLE_DEVICES=6

#!/bin/bash

# Define the tasks array
tasks=("sentiment" "topic" "multi")

# Loop through 1 to 7
for i in {8..9}
do
    # Loop through each task
    for task in "${tasks[@]}"
    do
        # Set the model path, output file, and other variables based on the loop index and task
        model_path="/data1/chh/models/model_sft/llama3-8b/merge/qwen_st/sft${i}"
        output="./results/sft_result/sft_st_${i}_${task}.json"
        mode="zero_shot"

        # Run the Python script with the current settings
        python standard_ctg.py --model_path="$model_path" --task="$task" --mode="$mode" --output="$output"
        python standard_eval.py --task="$task" --input_file="$output"
    done
done
