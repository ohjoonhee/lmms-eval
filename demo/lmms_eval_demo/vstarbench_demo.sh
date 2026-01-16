#!/bin/bash

uv run python demo/app.py $@ --hf-repo "ohjoonhee/vstar_bench" --hf-split "test" --hf-image-col "image" --port 5001