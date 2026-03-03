import os
import re

from loguru import logger as eval_logger

from lmms_eval.api.registry import METRIC_REGISTRY, register_metric
from lmms_eval.llm_judge import Request, ServerConfig, get_server

# Initialize LLM Judge
API_TYPE = os.getenv("API_TYPE", "openai")
MODEL_VERSION = os.getenv("MODEL_VERSION", "gpt-4-0613")

try:
    server_config = ServerConfig(model_name=MODEL_VERSION, temperature=0.0, max_tokens=1024)
    server = get_server(server_name=API_TYPE, config=server_config)
except Exception as e:
    eval_logger.warning(f"Failed to initialize LLM Judge: {e}. LLM Judge metric will fail if used.")
    server = None


if "countbenchqa_llm_judge" not in METRIC_REGISTRY:

    @register_metric(
        metric="countbenchqa_llm_judge",
        higher_is_better=True,
        output_type="generate_until",
        aggregation="mean",
    )
    def llm_judge_fn(items):
        return items


def countbenchqa_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def countbenchqa_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"]

    choices = list(range(2, 11))
    choices_text = "\n".join([f"{chr(ord('A') + i - 2)}. {i}" for i in choices])
    text = f"Question: {question}\nOptions:\n{choices_text}"

    if "pre_prompt" in lmms_eval_specific_kwargs:
        text = lmms_eval_specific_kwargs["pre_prompt"] + text
    if "post_prompt" in lmms_eval_specific_kwargs:
        text = text + lmms_eval_specific_kwargs["post_prompt"]
    return text


def countbenchqa_doc_to_messages(doc, lmms_eval_specific_kwargs=None):
    image = countbenchqa_doc_to_visual(doc)[0]
    text = countbenchqa_doc_to_text(doc, lmms_eval_specific_kwargs)
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": image},
                {"type": "text", "text": text},
            ],
        }
    ]
    return msgs


def countbenchqa_process_results(doc, results):
    pred = results[0]
    target = doc["number"]

    # Calculate target letter (2 -> A, 3 -> B, etc.)
    target_letter = chr(ord("A") + target - 2)

    # Normalize prediction
    pred_text = pred.strip()

    # Extract numbers
    numbers = re.findall(r"\d+", pred_text)

    # Extract candidate letters (A-I)
    # Looking for single letters that might be the answer
    # Matches: "A", "Answer: A", "A.", "(A)", "A "
    letters = re.findall(r"(?:^|\s|\(|:)([A-I])(?:\.|,|\)|\s|$)", pred_text, re.IGNORECASE)

    exact_match = 0.0

    # Check if target number is in prediction (last number heuristic often used, but here simpler is better for robust match)
    # If the target number is explicitly stated, it's likely correct unless negated (which exact match handles poorly anyway)
    if str(target) in numbers:
        exact_match = 1.0
    # Check if target letter is in prediction
    elif target_letter.upper() in [l.upper() for l in letters]:
        exact_match = 1.0

    # LLM Judge Logic
    llm_judge_score = 0.0
    if server:
        try:
            prompt = f"Question: {doc['question']}\nGround Truth: {target}\nPrediction: {pred}\nIs the prediction correct? Answer 'yes' or 'no'."
            request = Request(messages=[{"role": "user", "content": prompt}], config=server_config)
            response = server.evaluate(request)
            content = response.content.strip().lower()
            if "yes" in content:
                llm_judge_score = 1.0
        except Exception as e:
            eval_logger.error(f"LLM Judge error: {e}")

    return {
        "exact_match": exact_match,
        "llm_judge": llm_judge_score,
        "submission": {"question_id": str(doc.get("id", doc.get("image_id", hash(str(doc))))), "prediction": pred, "ground_truth": target},  # Fallback ID
    }


def countbenchqa_aggregate_exact_match(results):
    clean_results = [r[0] if isinstance(r, list) else r for r in results]
    return sum(clean_results) / len(results)


def countbenchqa_aggregate_llm_judge(results):
    clean_results = [r[0] if isinstance(r, list) else r for r in results]
    return sum(clean_results) / len(results)
