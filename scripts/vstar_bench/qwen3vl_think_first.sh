#!/bin/bash
set -a
source .env
set +a

# tasks
export TASKS="vstar_bench"
export RUN_NAME="$TASKS-think-first"

# LMMS EVAL cache configs
export LMMS_EVAL_USE_CACHE=True
export LMMS_EVAL_HOME="./tmp/lmms_eval_cache/$RUN_NAME"

# Judge model
export MODEL_VERSION="gpt-4.1-nano-2025-04-14"

export OUTPUT_DIR="output/$RUN_NAME"
# export LOGPROB_OUTPUT_DIR="$OUTPUT_DIR/logprob"

python3 -m accelerate.commands.launch \
    --num_processes=1 \
    -m lmms_eval \
    --model vllm \
    --model_args model="Qwen/Qwen3-VL-8B-Thinking",dtype=bfloat16,gpu_memory_utilization=0.90,max_model_len=32768,enforce_eager=True \
    --gen_kwargs max_new_tokens=4096,temperature=1.0,do_sample=True,top_p=0.95,top_k=20,repetition_penalty=1.0,presence_penalty=0.0 \
    --tasks $TASKS \
    --batch_size 64 \
    --log_samples \
    --output_path "$OUTPUT_DIR" \
    --lmms_eval_specific_kwargs "configs/prompts/think_first_answer_concise.yaml" \
    --wandb_log_samples \
    --wandb_args project=lmms-eval,job_type=eval,name="$RUN_NAME" \