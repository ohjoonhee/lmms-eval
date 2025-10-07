#!/bin/bash
set -a
source .env
set +a

# LMMS EVAL cache configs
export LMMS_EVAL_USE_CACHE=True
export LMMS_EVAL_HOME="./tmp/lmms_eval_cache_glm45"

# tasks
export TASKS="hrbench"
# Judge model
export MODEL_VERSION="gpt-4.1-nano-2025-04-14"

export OUTPUT_DIR="outputs/hrbench_glm45v_think"

python3 -m accelerate.commands.launch \
    --num_processes=1 \
    -m lmms_eval \
    --model vllm \
    --model_args model="zai-org/GLM-4.5V-FP8",gpu_memory_utilization=0.90,dtype=bfloat16,max_model_len=32768,cpu_offload_gb=32 \
    --gen_kwargs max_new_tokens=8192 \
    --tasks $TASKS \
    --batch_size 64 \
    --log_samples \
    --output_path "$OUTPUT_DIR" \
    --wandb_log_samples \
    --wandb_args project=lmms-eval,job_type=eval,name="$(basename $OUTPUT_DIR)" 
    # --limit 8 \
