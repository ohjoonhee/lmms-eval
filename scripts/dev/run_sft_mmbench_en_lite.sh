#!/bin/bash
# Generic wrapper to run evaluations with the SFT model
# Usage: ./run_sft_eval.sh <task_name> [batch_size]
# Example: ./run_sft_eval.sh mmstar 16

set -a
source .env
set +a

TASKS="mmbench_en_dev_lite"
BATCH_SIZE=8

export RUN_NAME="mmbench-en-lite-sft"
export LMMS_EVAL_USE_CACHE=True
export LMMS_EVAL_HOME="./tmp/lmms_eval_cache/$RUN_NAME"
export MODEL_VERSION="gpt-4.1-nano-2025-04-14"
export OUTPUT_DIR="output/$RUN_NAME"

python3 -m accelerate.commands.launch \
    --num_processes=1 \
    -m lmms_eval \
    --model vllm \
    --model_args model="ohjoonhee/Qwen3-VL-4B-Thinking-Viscot46k-SFT-v1",dtype=bfloat16,gpu_memory_utilization=0.90,max_model_len=32768 \
    --gen_kwargs max_new_tokens=8192,temperature=1.0,do_sample=True,top_p=0.95,top_k=20,repetition_penalty=1.0,presence_penalty=0.0 \
    --tasks "$TASKS" \
    --batch_size "$BATCH_SIZE" \
    --log_samples \
    --output_path "$OUTPUT_DIR"