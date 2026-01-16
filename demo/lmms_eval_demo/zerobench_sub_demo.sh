#!/bin/bash

uv run python demo/app.py $@ --hf-repo "jonathan-roberts1/zerobench" --hf-split "zerobench_subquestions" --hf-image-col "question_images_decoded" --port 5001