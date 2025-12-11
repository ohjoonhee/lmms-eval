#!/bin/bash

vllm serve Qwen/Qwen3-VL-8B-Thinking \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --gpu-memory-utilization 0.95 \
    --enforce-eager \
    --max-model-len 32768 \
    # --quantization fp8
    # --reasoning-parser deepseek_r1 \