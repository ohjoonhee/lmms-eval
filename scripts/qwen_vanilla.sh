set -a
source .env
set +a

# LMMS EVAL cache configs
export LMMS_EVAL_USE_CACHE=True
export LMMS_EVAL_HOME="./tmp/lmms_eval_cache/qwen_vanilla"

# Output directory and tasks
export TASKS="hrbench"
export OUTPUT_DIR="outputs/hrbench_qwen_vanilla"

# Judge config
export MODEL_VERSION="gpt-4.1-nano-2025-04-14"



python3 -m accelerate.commands.launch \
    --num_processes=1 \
    -m lmms_eval \
    --model qwen3_vl \
    --tasks $TASKS \
    --log_samples \
    --batch_size 1 \
    --output_path "$OUTPUT_DIR" \
    --wandb_log_samples \
    --wandb_args project=lmms-eval,job_type=eval,name="$(basename $OUTPUT_DIR)" 
    # --limit 8 \

    # --model_args model="Qwen/Qwen2.5-VL-3B-Instruct",gpu_memory_utilization=0.90,dtype=bfloat16,max_model_len=32768 \