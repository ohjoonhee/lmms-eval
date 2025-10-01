set -a
source .env
set +a

# LMMS EVAL cache configs
export LMMS_EVAL_USE_CACHE=True
export LMMS_EVAL_HOME="./tmp/lmms_eval_cache"

# Output directory and tasks
export TASKS="vstar_bench"
export OUTPUT_DIR="outputs/"$TASKS"_intern35_1b-think"


export MODEL_VERSION="gpt-5-nano-2025-08-07"

python3 -m accelerate.commands.launch \
    --num_processes=1 \
    -m lmms_eval \
    --model vllm \
    --model_args model="OpenGVLab/InternVL3_5-1B",gpu_memory_utilization=0.90,dtype=bfloat16,max_model_len=32768 \
    --tasks $TASKS \
    --batch_size 4 \
    --log_samples \
    --output_path "$OUTPUT_DIR" \
    --wandb_log_samples \
    --wandb_args project=lmms-eval,job_type=eval,name="$(basename $OUTPUT_DIR)" 
    # --limit 8 \
    # --verbosity DEBUG \

export OUTPUT_DIR="outputs/"$TASKS"_intern35_4b-think"

python3 -m accelerate.commands.launch \
    --num_processes=1 \
    -m lmms_eval \
    --model vllm \
    --model_args model="OpenGVLab/InternVL3_5-4B",gpu_memory_utilization=0.90,dtype=bfloat16,max_model_len=32768 \
    --tasks $TASKS \
    --batch_size 64 \
    --log_samples \
    --output_path "$OUTPUT_DIR" \
    --wandb_log_samples \
    --wandb_args project=lmms-eval,job_type=eval,name="$(basename $OUTPUT_DIR)" 

export OUTPUT_DIR="outputs/"$TASKS"_intern35_2b-think"

python3 -m accelerate.commands.launch \
    --num_processes=1 \
    -m lmms_eval \
    --model vllm \
    --model_args model="OpenGVLab/InternVL3_5-2B",gpu_memory_utilization=0.90,dtype=bfloat16,max_model_len=32768 \
    --tasks $TASKS \
    --batch_size 64 \
    --log_samples \
    --output_path "$OUTPUT_DIR" \
    --wandb_log_samples \
    --wandb_args project=lmms-eval,job_type=eval,name="$(basename $OUTPUT_DIR)" 

export OUTPUT_DIR="outputs/"$TASKS"_intern35_8b-think"

python3 -m accelerate.commands.launch \
    --num_processes=1 \
    -m lmms_eval \
    --model vllm \
    --model_args model="OpenGVLab/InternVL3_5-8B",gpu_memory_utilization=0.90,dtype=bfloat16,max_model_len=32768 \
    --tasks $TASKS \
    --batch_size 64 \
    --log_samples \
    --output_path "$OUTPUT_DIR" \
    --wandb_log_samples \
    --wandb_args project=lmms-eval,job_type=eval,name="$(basename $OUTPUT_DIR)" 
