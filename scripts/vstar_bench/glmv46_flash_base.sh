#!/bin/bash
set -a
source .env
set +a

# tasks
export TASKS="hrbench"
export MODEL="zai-org/GLM-4.6V-Flash"

export RUN_NAME="$TASKS-r1-think-sysprompt"

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
    --model_args model="$MODEL",dtype=bfloat16,gpu_memory_utilization=0.90,max_model_len=32768,reasoning_parser="glm45" \
    --gen_kwargs max_new_tokens=16384,top_p=0.6,top_k=2,temperature=0.8,repetition_penalty=1.1 \
    --tasks $TASKS \
    --batch_size 32 \
    --log_samples \
    --output_path "$OUTPUT_DIR" \
    --wandb_log_samples \
    --wandb_args project=lmms-eval,job_type=eval,name="$RUN_NAME" \
    --lmms_eval_specific_kwargs "configs/prompts/r1_think_system.yaml" \
    # --gen_kwargs max_new_tokens=4096,temperature=1.0,do_sample=True,top_p=0.95,top_k=20,repetition_penalty=1.0,presence_penalty=0.0 \