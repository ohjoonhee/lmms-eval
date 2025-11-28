#!/bin/bash
set -a
source .env
set +a

# LMMS EVAL cache configs
export LMMS_EVAL_USE_CACHE=True
export LMMS_EVAL_HOME="./tmp/lmms_eval_cache/qwen3_brain_eye_eyeparam"

# tasks
export TASKS="hrbench"
# Judge model
export MODEL_VERSION="gpt-4.1-nano-2025-04-14"

export OUTPUT_DIR="outputs/hrbench-qwen3-brain-eye-eyeparam"

accelerate launch --num_processes=1 -m lmms_eval \
    --model qwen3_brain_eye_vllm \
    --model_args "brain_model=Qwen/Qwen3-4B,eye_model=Qwen/Qwen3-VL-2B-Instruct,gpu_memory_utilization=0.4" \
    --tasks hrbench \
    --batch_size 1 \
    --log_samples \
    --output_path "$OUTPUT_DIR" \
