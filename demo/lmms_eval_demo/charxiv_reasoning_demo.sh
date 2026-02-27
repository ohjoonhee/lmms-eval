#!/bin/bash

uv run python demo/lmms_eval_demo/app.py $@ --hf-repo "princeton-nlp/CharXiv" --hf-split "validation" --hf-image-col "image" --port 5001
