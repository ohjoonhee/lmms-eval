set -a
source .env
set +a

# LMMS EVAL cache configs
export LMMS_EVAL_USE_CACHE=True
export LMMS_EVAL_HOME="./tmp/lmms_eval_cache"

# Output directory and tasks
export OUTPUT_DIR="outputs/hrbench_qwen25_3b-think-gpt5nanojudge"
export TASKS="hrbench"

export MODEL_VERSION="gpt-5-nano-2025-08-07"



python3 -m accelerate.commands.launch \
    --num_processes=1 \
    -m lmms_eval \
    --model vllm \
    --model_args model="Qwen/Qwen2.5-VL-3B-Instruct",gpu_memory_utilization=0.90,dtype=bfloat16,max_model_len=32768 \
    --tasks $TASKS \
    --batch_size 64 \
    --log_samples \
    --output_path "$OUTPUT_DIR" \
    --wandb_log_samples \
    --wandb_args project=lmms-eval,job_type=eval,name="$(basename $OUTPUT_DIR)" 
    # --limit 8 \

export OUTPUT_DIR="outputs/hrbench_qwen25_7b-think-gpt5nanojudge"

python3 -m accelerate.commands.launch \
    --num_processes=1 \
    -m lmms_eval \
    --model vllm \
    --model_args model="Qwen/Qwen2.5-VL-7B-Instruct",gpu_memory_utilization=0.90,dtype=bfloat16,max_model_len=32768 \
    --tasks $TASKS \
    --batch_size 64 \
    --log_samples \
    --output_path "$OUTPUT_DIR" \
    --wandb_log_samples \
    --wandb_args project=lmms-eval,job_type=eval,name="$(basename $OUTPUT_DIR)" 
    # --limit 8 \
