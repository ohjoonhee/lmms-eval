#!/bin/bash
#SBATCH -J launch
#SBATCH -p gpu-4farm
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH -o logs/%x_%j.log
#SBATCH --account=gpu

set -x

export OUTPUT_DIR="outputs/hrbench_glm45v"
export TASKS="hrbench"

python3 -m accelerate.commands.launch \
    --num_processes=1 \
    --num_cpu_threads_per_process=28 \
    -m lmms_eval \
    --model vllm \
    --model_args model="zai-org/GLM-4.5V-FP8",gpu_memory_utilization=0.95,dtype=bfloat16,max_model_len=8192,tensor_parallel_size=4 \
    --tasks $TASKS \
    --batch_size 64 \
    --log_samples \
    --output_path "$OUTPUT_DIR" \
    --wandb_log_samples \
    --wandb_args project=lmms-eval,job_type=eval,name="$(basename $OUTPUT_DIR)" 
