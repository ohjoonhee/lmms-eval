import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from loguru import logger as eval_logger

from lmms_eval.llm_judge import Request, ServerConfig, get_server

with open(Path(__file__).parent / "_default_template_yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

    config = yaml.safe_load("".join(safe_data))

API_TYPE = os.getenv("API_TYPE", "openai")
MODEL_VERSION = os.getenv("MODEL_VERSION", "gpt-4o-2024-11-20")

server_config = ServerConfig(model_name=MODEL_VERSION, temperature=0.0, max_tokens=128)
server = get_server(server_name=API_TYPE, config=server_config)


def _extract_answer_letter(text: str) -> str:
    """
    Extract the answer choice letter from a string.

    Examples:
    'A answer1' -> 'A'
    'A) answer2' -> 'A'
    '(B) answer' -> 'B'
    'C' -> 'C'
    '(C)' -> 'C'
    'A.' -> 'A'

    Return an empty string if no letter is found.
    """
    text = text.strip()
    match = re.match(r"[\(\s]*([A-Z])[\)\.\s]*", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return ""


def blink_doc_to_text(doc: dict[str, Any], lmms_eval_specific_kwargs: Optional[dict[str, Any]] = None) -> str:
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    options_labels = ["A", "B", "C", "D", "E"]
    num_options = len(doc["choices"])
    options_current_task = ", ".join(options_labels[:num_options])
    prompt = lmms_eval_specific_kwargs.get("pre_prompt", "").format(options_current_task) + doc["prompt"]
    return prompt


def blink_doc_to_visual(doc: dict) -> list:
    keys = doc.keys()
    image_keys = [item for item in keys if re.match(r"^image_\d+$", item)]
    image_list = []
    for image_key in image_keys:
        image = doc[image_key]
        if image is not None:
            image_list.append(image.convert("RGB"))
    return image_list


def _get_judge_response(
    prompt: str,
    model: str = MODEL_VERSION,
    temperature: float = 0.0,
    max_tokens: int = 128,
    patience: int = 3,
    sleep_time: float = 5.0,
) -> tuple[str, str]:
    custom_config = ServerConfig(
        model_name=model, temperature=temperature, max_tokens=max_tokens
    )

    while patience > 0:
        patience -= 1
        try:
            request = Request(
                messages=[{"role": "user", "content": prompt}], config=custom_config
            )
            response = server.evaluate(request)
            content = response.content.strip() if response.content else ""
            if content != "":
                return content, response.model_used
        except Exception as e:
            eval_logger.error(f"Error: {e}")
            if "Rate limit" in str(e):
                eval_logger.info("Sleeping due to rate limit...")
                time.sleep(sleep_time)
            eval_logger.info(f"Retrying...Patience left: {patience}")

    return "", ""


def _build_judge_prompt(doc: Dict, response: str) -> str:
    """Build the prompt for the LLM judge to evaluate a BLINK response."""
    question = doc["prompt"]
    ground_truth = doc["answer"].strip("()")

    # Build option list for context
    choices = doc["choices"]
    option_labels = ["A", "B", "C", "D", "E"]
    options_text = "\n".join(
        f"({label}) {choice}" for label, choice in zip(option_labels[: len(choices)], choices)
    )

    return (
        "You are a strict evaluator for a multiple-choice visual question answering task. "
        "Determine whether the model's prediction matches the correct answer.\n\n"
        f"Question:\n{question}\n\n"
        f"Options:\n{options_text}\n\n"
        f"Correct Answer: ({ground_truth})\n\n"
        f"Model Prediction:\n{response}\n\n"
        "Rules:\n"
        "- The model prediction may contain reasoning. Focus on the final answer.\n"
        "- Score 1 if the predicted answer matches the correct option letter or its content.\n"
        "- Score 0 for incorrect, ambiguous, or missing answers.\n"
        "- Ignore minor formatting differences.\n\n"
        "Output only a single number: 1 or 0"
    )


def blink_process_results(doc: Dict, result: List[str]) -> Dict[str, Dict]:
    # extract grounded answer
    grounded_output = doc["answer"].strip("()")
    response = result[0]

    # extract predicted answer (parsing-based)
    pred_letter = _extract_answer_letter(response)
    flag = pred_letter == grounded_output

    submission = {
        "id": doc["idx"],
        "gt_content": grounded_output,
        "pred_parsed": pred_letter,
        "pred": response,
        "sub_task": doc["sub_task"],
        "is_correct": flag,
    }

    # LLM judge evaluation
    judge_prompt = _build_judge_prompt(doc, response)
    judge_score = 0.0
    judge_content, judge_model = _get_judge_response(judge_prompt)
    if judge_content:
        try:
            score = int(judge_content.strip().split()[0])
            if score in (0, 1):
                judge_score = float(score)
            else:
                eval_logger.warning(
                    f"Unexpected judge score '{judge_content}' for doc {doc['idx']}, defaulting to 0."
                )
        except (ValueError, IndexError):
            eval_logger.warning(
                f"Failed to parse judge response '{judge_content}' for doc {doc['idx']}, defaulting to 0."
            )
    else:
        eval_logger.warning(f"Empty judge response for doc {doc['idx']}, defaulting to 0.")

    judge_submission = {
        "id": doc["idx"],
        "gt_content": grounded_output,
        "pred": response,
        "sub_task": doc["sub_task"],
        "score": judge_score,
        "eval_model": judge_model,
    }

    return {"blink_acc": submission, "blink_gpt_eval": judge_submission}


def blink_aggregate_results(results: List[Dict]) -> float:
    total_samples = len(results)
    total_correct = 0

    for sample in results:
        if sample["is_correct"]:
            total_correct += 1

    accuracy = total_correct / total_samples if total_samples > 0 else 0
    return accuracy


def blink_aggregate_gpt_eval(results: List[Dict]) -> float:
    """Aggregate LLM judge scores as mean accuracy."""
    if not results:
        return 0.0
    total_score = sum(r["score"] for r in results)
    return total_score / len(results)
