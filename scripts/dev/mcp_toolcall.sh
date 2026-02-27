#!/bin/bash
set -a
source .env
set +a

# --- Configuration ---
# tasks
export TASKS="hrbench"
export RUN_NAME="dev-$TASKS-example-mcp"

# LMMS EVAL cache configs
export LMMS_EVAL_USE_CACHE=True
export LMMS_EVAL_HOME="./tmp/lmms_eval_cache/$RUN_NAME"

# Judge model
export MODEL_VERSION="gpt-4.1-nano-2025-04-14"

export OUTPUT_DIR="output/$RUN_NAME"
# export LOGPROB_OUTPUT_DIR="$OUTPUT_DIR/logprob"

# Serve vllm text thinking model
export MODEL="Qwen/Qwen3-VL-2B-Instruct"
export PORT=10630
export VLLM_LOG_FILE="$OUTPUT_DIR/vllm_server.log"
mkdir -p "$OUTPUT_DIR"


# Check if port is already in use (simple check, not blocking if curl fails)
if curl -s "http://localhost:$PORT/health" > /dev/null; then
    echo "Warning: Port $PORT seems to be already in use. Attempting to restart/use existing?"
    # Ideally we might want to fail or kill, but for now just warn.
fi

# --- Start vLLM ---
echo "Starting vLLM server..."
echo "Logs will be written to $VLLM_LOG_FILE"

uv run vllm serve $MODEL \
    --port $PORT \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --gpu-memory-utilization 0.95 \
    --enforce-eager \
    --max-model-len 65536 \
    > "$VLLM_LOG_FILE" 2>&1 &

# Save the Process ID (PID) of the background job
VLLM_PID=$!
echo "VLLM server started with PID: $VLLM_PID"

# --- Signal Handling ---
# Trap to ensure cleanup on exit or interruption
cleanup() {
    echo ""
    echo "Stopping VLLM server (PID: $VLLM_PID)..."
    if kill -0 $VLLM_PID 2>/dev/null; then
        kill $VLLM_PID
    fi
    exit 0
}
trap cleanup EXIT SIGINT SIGTERM

# --- Wait for Server ---
# A more robust approach is to use a health check loop.
SERVER_READY=0
MAX_ATTEMPTS=60 # 60 * 5s = 5 minutes
ATTEMPT=0
echo "Checking VLLM server readiness..."

while [ $SERVER_READY -eq 0 ] && [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
    # Check if process is still alive
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "ERROR: VLLM server process died unexpectedly."
        echo "Last 20 lines of log:"
        tail -n 20 "$VLLM_LOG_FILE"
        exit 1
    fi

    if curl -s "http://localhost:$PORT/health" > /dev/null; then
        SERVER_READY=1
        echo "VLLM server is READY."
    else
        ATTEMPT=$((ATTEMPT + 1))
        echo "Attempt $ATTEMPT/$MAX_ATTEMPTS: Server not ready yet. Waiting 5 seconds..."
        sleep 5
    fi
done

if [ $SERVER_READY -eq 0 ]; then
    echo "ERROR: VLLM server failed to start or become ready after $MAX_ATTEMPTS attempts."
    echo "Check $VLLM_LOG_FILE for details."
    exit 1
fi

# --- Run Inference ---
echo "VLLM is ready. Running inference..."

python3 -m accelerate.commands.launch \
    --num_processes=1 \
    -m lmms_eval \
    --model async_openai \
    --model_args model_version="$MODEL",base_url="http://localhost:$PORT/v1",api_key="EMPTY",mcp_server_path="examples/mcp_server/sample_mcp_server.py",is_qwen3_vl=True \
    --gen_kwargs max_new_tokens=4096,temperature=0.7,do_sample=True,top_p=0.8,top_k=20,repetition_penalty=1.0,presence_penalty=1.5 \
    --tasks $TASKS \
    --batch_size 16 \
    --log_samples \
    --output_path "$OUTPUT_DIR" \
    --verbosity WARNING \
    # --wandb_log_samples \
    # --wandb_args project=lmms-eval,job_type=eval,name="$RUN_NAME" \

echo "Inference complete."