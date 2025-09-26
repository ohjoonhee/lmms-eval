#!/bin/bash
#SBATCH -J launch
#SBATCH -p gpu-1farm
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH -o logs/%x_%j.log
#SBATCH --time=1-00:00:00
#SBATCH --account=gpu

set -x

export OUTPUT_DIR="outputs/hrbench_glm45v"
export TASKS="hrbench"

python3 -m accelerate.commands.launch \
    --num_processes=1 \
    -m lmms_eval \
    --model vllm \
    --model_args model="zai-org/GLM-4.5V-FP8",gpu_memory_utilization=0.90,dtype=bfloat16,max_model_len=32768 \
    --tasks $TASKS \
    --batch_size 64 \
    --log_samples \
    --output_path "$OUTPUT_DIR" \
    --wandb_log_samples \
    --wandb_args project=lmms-eval-rlvr,job_type=eval,name="$(basename $OUTPUT_DIR)" 
    # --limit 8 \
