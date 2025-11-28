#!/bin/bash
set -a
source .env
set +a

# LMMS EVAL cache configs
export LMMS_EVAL_USE_CACHE=True
export LMMS_EVAL_HOME="./tmp/lmms_eval_cache/qwen3vl_think_logprob"

# tasks
export TASKS="hrbench"
# Judge model
export MODEL_VERSION="gpt-4.1-nano-2025-04-14"

export OUTPUT_DIR="outputs/hrbench-qwen3vl-30ba3b-think-fp8-logprob"
export LOGPROB_OUTPUT_DIR="$OUTPUT_DIR/logprob"

uv run python3 -m accelerate.commands.launch \
    --num_processes=1 \
    -m lmms_eval \
    --model vllm \
    --model_args model="Qwen/Qwen3-VL-30B-A3B-Thinking-FP8",gpu_memory_utilization=0.90,max_model_len=32768,quantization=fp8,enforce_eager=True \
    --gen_kwargs max_new_tokens=4096,temperature=0.6,do_sample=True \
    --tasks $TASKS \
    --batch_size 16 \
    --log_samples \
    --output_path "$OUTPUT_DIR" \
    --wandb_log_samples \
    --wandb_args project=lmms-eval,job_type=eval,name="$(basename $OUTPUT_DIR)" 
    # --limit 8 \
