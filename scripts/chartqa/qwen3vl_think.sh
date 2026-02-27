#!/bin/bash
set -a
source .env
set +a

# tasks
export TASKS="chartqa_lite"
export RUN_NAME="$TASKS-base"

# LMMS EVAL cache configs
export LMMS_EVAL_USE_CACHE=True
export LMMS_EVAL_HOME="./tmp/lmms_eval_cache/$RUN_NAME"

export OUTPUT_DIR="output/$RUN_NAME"

uv run python3 -m accelerate.commands.launch \
    --num_processes=1 \
    -m lmms_eval \
    --model vllm \
    --model_args model="Qwen/Qwen3-VL-4B-Thinking",dtype=bfloat16,gpu_memory_utilization=0.90,max_model_len=32768,disable_log_stats=True \
    --gen_kwargs max_new_tokens=4096,temperature=1.0,do_sample=True,top_p=0.95,top_k=20,repetition_penalty=1.0,presence_penalty=0.0 \
    --tasks $TASKS \
    --batch_size 16 \
    --log_samples \
    --output_path "$OUTPUT_DIR" \
