#!/bin/bash

vllm serve Qwen/Qwen3-VL-2B-Instruct \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --gpu-memory-utilization 0.5 \
    --max-model-len 65536 \
    # --max-model-len 131072 \
    # --quantization fp8
    # --reasoning-parser deepseek_r1 \