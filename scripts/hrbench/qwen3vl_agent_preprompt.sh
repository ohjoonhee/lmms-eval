#!/bin/bash
set -a
source .env
set +a

export OMP_NUM_THREADS=8
export VLLM_CPU_OMP_THREADS_BIND=0-7

# tasks
export TASKS="hrbench"
export RUN_NAME="$TASKS-base-preprompt"

# LMMS EVAL cache configs
export LMMS_EVAL_USE_CACHE=True
export LMMS_EVAL_HOME="./tmp/lmms_eval_cache/$RUN_NAME"

# Judge model
export MODEL_VERSION="gpt-4.1-nano-2025-04-14"

export OUTPUT_DIR="output/$RUN_NAME"
# export LOGPROB_OUTPUT_DIR="$OUTPUT_DIR/logprob"

# Serve vLLM model locally
# Define the log file path
VLLM_LOG_FILE="$OUTPUT_DIR/vllm_serve/$(date +%Y-%m-%d_%H-%M-%S).log"

# Ensure the output directory exists before redirection
mkdir -p "$(dirname $VLLM_LOG_FILE)"

export PORT=10630
export BASE_URL="http://localhost:$PORT/v1"
export API_KEY="EMPTY"
export MODEL="Qwen/Qwen3-VL-2B-Instruct"

vllm serve $MODEL \
    --port $PORT \
    --api-key $API_KEY \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --gpu-memory-utilization 0.95 \
    --max_num_seq 1 \
    --max-model-len 131072 > "$VLLM_LOG_FILE" 2>&1 &
    # --reasoning-parser deepseek_r1 &
    # --quantization fp8 &
    # --tensor-parallel-size 4 \

# Save the Process ID (PID) of the background job
VLLM_PID=$!
echo "VLLM server started with PID: $VLLM_PID"

# A more robust approach is to use a health check loop.
# This loop checks a port/endpoint until the server is ready.
# This assumes vllm is serving on localhost:$PORT and has a /health endpoint.
SERVER_READY=0
MAX_ATTEMPTS=30
ATTEMPT=0
echo "Checking VLLM server readiness..."
while [ $SERVER_READY -eq 0 ] && [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
    if curl -s http://localhost:$PORT/health > /dev/null; then
        SERVER_READY=1
        echo "VLLM server is READY."
    else
        ATTEMPT=$((ATTEMPT + 1))
        echo "Attempt $ATTEMPT/$MAX_ATTEMPTS: Server not ready yet. Waiting 30 seconds..."
        sleep 30
    fi
done

if [ $SERVER_READY -eq 0 ]; then
    echo "ERROR: VLLM server failed to start or become ready after $MAX_ATTEMPTS attempts."
    # Kill the background process and exit the script
    kill $VLLM_PID
    exit 1
fi

echo "VLLM is ready. Running inference script..."

python3 -m accelerate.commands.launch \
    --num_processes=1 \
    -m lmms_eval \
    --model qwen_agent \
    --model_args model_version="$MODEL",base_url="$BASE_URL",api_key="$API_KEY" \
    --gen_kwargs max_new_tokens=4096,temperature=0.7,do_sample=True,top_p=0.8,top_k=20,repetition_penalty=1.0,presence_penalty=1.5 \
    --tasks $TASKS \
    --batch_size 1 \
    --log_samples \
    --output_path "$OUTPUT_DIR" \
    --lmms_eval_specific_kwargs "configs/prompts/qwen3vl_hrbench_pre.yaml"

echo "Inference complete. Stopping VLLM server (PID: $VLLM_PID)..."
kill $VLLM_PID

echo "Script finished."