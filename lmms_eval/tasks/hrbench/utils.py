import base64
import datetime
import io
import json
import os
import string
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from loguru import logger as eval_logger
from PIL import Image

from lmms_eval.tasks.hrbench.hrbench_evals import HRBenchEval

with open(Path(__file__).parent / "hrbench.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

    config = yaml.safe_load("".join(safe_data))

hrbench_evaluator = HRBenchEval(api_key=os.getenv("OPENAI_API_KEY", "YOUR_API_KEY"), gpt_model=os.getenv("MODEL_VERSION", "gpt-4o-2024-11-20"), max_workers=config["metadata"]["max_workers"])

THINKING_SYSTEM_PROMPT = """
You are an AI assistant that rigorously follows this response protocol:

1. First, conduct a detailed analysis of the question. Consider different angles, potential solutions, and reason through the problem step-by-step. Enclose this entire thinking process within <think> and </think> tags.

2. After the thinking section, provide a clear, concise, and direct answer to the user's question. Separate the answer from the think section with a newline.

Ensure that the thinking process is thorough but remains focused on the query. The final answer should be standalone and not reference the thinking section.
""".strip()


def decode_base64_to_image(base64_string, target_size=-1):
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    if target_size > 0:
        image.thumbnail((target_size, target_size))
    return image


def hrbench_doc_to_visual(doc):
    image = decode_base64_to_image(doc["image"])
    return [image]


def hrbench_doc_to_options(doc):
    options = {cand: doc[cand] for cand in string.ascii_uppercase if cand in doc and not pd.isna(doc[cand])}
    return options


def hrbench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"].strip()
    options = hrbench_doc_to_options(doc)
    options_prompt = ""
    for key, item in options.items():
        options_prompt += f"{key}. {item}\n"
    prompt = ""
    # prompt += f"{question}\n{options_prompt}Answer the option letter directly."
    prompt += f"{question}\n{options_prompt}"
    return prompt


def hrbench_doc_to_messages(doc):
    """
    Convert a document to a list of messages with proper typing.
    Supports interleaved text, images, videos, and audio.
    """
    messages = []

    # Add system message if needed
    messages.append({"role": "system", "content": [{"type": "text", "text": THINKING_SYSTEM_PROMPT}]})

    # Add user message with multimodal content
    image = hrbench_doc_to_visual(doc)[0]
    user_content = []
    user_content.append({"type": "image", "url": image})

    prompt = hrbench_doc_to_text(doc)
    user_content.append({"type": "text", "text": prompt})

    messages.append({"role": "user", "content": user_content})

    return messages


def hrbench_process_results(doc, results):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name, value: metric value
    """
    pred = results[0].strip()
    gt = doc["answer"]
    options = hrbench_doc_to_options(doc)
    question = doc["question"]
    resp_dic = hrbench_evaluator.get_chat_response({"question": question, "options": options, "prediction": pred})
    gpt_prediction = resp_dic["gpt_prediction"]
    category = doc["category"]
    cycle_category = doc["cycle_category"]

    gpt_score = 0
    if gt.lower() == gpt_prediction.lower():
        gpt_score = 1

    return {category: {"index": doc["index"], "cycle_category": cycle_category, "gpt_score": gpt_score}, "average": {"index": doc["index"], "cycle_category": cycle_category, "gpt_score": gpt_score}}


def hrbench_aggregate_results(results, args):
    """
    Args:
        results: a list of values returned by process_results
    Returns:
        A score
    """
    cycle_category_scores = defaultdict(list)
    for result in results:
        score = result["gpt_score"]
        cycle_category = result["cycle_category"]
        cycle_category_scores[cycle_category].append(score)

    cycle_category_avg_score = {}
    for cycle_category, scores in cycle_category_scores.items():
        avg_score = sum(scores) / len(scores)
        cycle_category_avg_score[cycle_category] = avg_score

    avg_score = sum(cycle_category_avg_score.values()) / len(cycle_category_avg_score)
    return avg_score
