export OMP_NUM_THREADS=12

export OUTPUT_DIR="outputs/dev-hrbench_intern35"
export TASKS="hrbench"

# python3 -m accelerate.commands.launch \
#     --num_processes=1 \
#     -m lmms_eval \
#     --model vllm \
#     --model_args model_version="Qwen/Qwen2.5-VL-7B-Instruct",gpu_memory_utilization=0.80,dtype=bfloat16,max_model_len=16384 \
#     --tasks $TASKS \
#     --batch_size 64 \
#     --log_samples \
#     --output_path "$OUTPUT_DIR/base" \
#     --wandb_log_samples \
#     --wandb_args project=lmms-eval-rlvr,job_type=eval,name="$(basename $OUTPUT_DIR)" 

python3 -m accelerate.commands.launch \
    --num_processes=1 \
    -m lmms_eval \
    --model vllm \
    --model_args model="OpenGVLab/InternVL3_5-4B",gpu_memory_utilization=0.90,dtype=bfloat16,max_model_len=32768 \
    --tasks $TASKS \
    --batch_size 4 \
    --log_samples \
    --output_path "$OUTPUT_DIR" \
    --limit 8 \
    # --wandb_log_samples \
    # --wandb_args project=lmms-eval-rlvr,job_type=eval,name="$(basename $OUTPUT_DIR)" 