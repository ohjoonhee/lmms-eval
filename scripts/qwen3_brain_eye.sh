#!/bin/bash
set -a
source .env
set +a


# tasks
export TASKS="zerobench"
# Judge model
export MODEL_VERSION="gpt-4.1-nano-2025-04-14"
export OPENAI_API_URL="https://api.openai.com/v1"

export OUTPUT_DIR="outputs/$TASKS-qwen3-brain-eye-eyeparam"

# LMMS EVAL cache configs
export LMMS_EVAL_USE_CACHE=True
export LMMS_EVAL_HOME="./tmp/lmms_eval_cache/$TASKS-qwen3-brain-eye-eyeparam"

# Serve vllm text thinking model
export TEXT_MODEL="Qwen/Qwen3-4B"

vllm serve $TEXT_MODEL --enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser deepseek_r1 --gpu-memory-utilization 0.4 &

# Save the Process ID (PID) of the background job
VLLM_PID=$!
echo "VLLM server started with PID: $VLLM_PID"

# A more robust approach is to use a health check loop.
# This loop checks a port/endpoint until the server is ready.
# This assumes vllm is serving on localhost:8000 and has a /health endpoint.
SERVER_READY=0
MAX_ATTEMPTS=30
ATTEMPT=0
echo "Checking VLLM server readiness..."
while [ $SERVER_READY -eq 0 ] && [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
    if curl -s http://localhost:8000/health > /dev/null; then
        SERVER_READY=1
        echo "VLLM server is READY."
    else
        ATTEMPT=$((ATTEMPT + 1))
        echo "Attempt $ATTEMPT/$MAX_ATTEMPTS: Server not ready yet. Waiting 2 seconds..."
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
accelerate launch --num_processes=1 -m lmms_eval \
    --model qwen3_brain_eye_vllm \
    --model_args "brain_model=$TEXT_MODEL,eye_model=Qwen/Qwen3-VL-2B-Instruct,gpu_memory_utilization=0.4" \
    --tasks $TASKS \
    --batch_size 1\
    --log_samples \
    --output_path "$OUTPUT_DIR" 

echo "Inference complete. Stopping VLLM server (PID: $VLLM_PID)..."
kill $VLLM_PID

echo "Script finished."