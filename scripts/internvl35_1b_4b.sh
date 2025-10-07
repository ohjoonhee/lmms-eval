#!/bin/bash
set -a
source .env
set +a

# LMMS EVAL cache configs
export LMMS_EVAL_USE_CACHE=True
export LMMS_EVAL_HOME="./tmp/lmms_eval_cache_maxnewtokens8k"

# tasks
export TASKS="hrbench"
# Judge model
export MODEL_VERSION="gpt-4.1-nano-2025-04-14"

export OUTPUT_DIR="outputs/hrbench_intern35_1b-think-maxnewtokens8k"

python3 -m accelerate.commands.launch \
    --num_processes=1 \
    -m lmms_eval \
    --model vllm \
    --model_args model="OpenGVLab/InternVL3_5-1B",gpu_memory_utilization=0.90,dtype=bfloat16,max_model_len=32768 \
    --gen_kwargs max_new_tokens=8192 \
    --tasks $TASKS \
    --batch_size 64 \
    --log_samples \
    --output_path "$OUTPUT_DIR" \
    --wandb_log_samples \
    --wandb_args project=lmms-eval,job_type=eval,name="$(basename $OUTPUT_DIR)" 
    # --limit 8 \

export OUTPUT_DIR="outputs/hrbench_intern35_4b-think-maxnewtokens8k"

python3 -m accelerate.commands.launch \
    --num_processes=1 \
    -m lmms_eval \
    --model vllm \
    --model_args model="OpenGVLab/InternVL3_5-4B",gpu_memory_utilization=0.90,dtype=bfloat16,max_model_len=32768 \
    --gen_kwargs max_new_tokens=8192 \
    --tasks $TASKS \
    --batch_size 64 \
    --log_samples \
    --output_path "$OUTPUT_DIR" \
    --wandb_log_samples \
    --wandb_args project=lmms-eval,job_type=eval,name="$(basename $OUTPUT_DIR)" 
    # --limit 8 \
