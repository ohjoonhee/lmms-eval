#!/bin/bash

uv run python demo/app.py $@ --hf-repo "ohjoonhee/HR-Bench" --hf-split "hrbench_4k" --hf-image-col "image" --port 5001