#!/bin/bash

uv run python demo/lmms_eval_demo/app.py $@ --hf-repo "moonshotai/WorldVQA" --hf-split "train" --hf-image-col "image" --port 5001
