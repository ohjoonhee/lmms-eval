#!/bin/bash

uv run python demo/lmms_eval_demo/app.py $@ --hf-repo "Lin-Chen/MMStar" --hf-split "val" --hf-image-col "image" --port 5001
