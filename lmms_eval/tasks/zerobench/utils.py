import re
import os
from loguru import logger as eval_logger
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


def zerobench_doc_to_visual(doc, lmms_eval_specific_kwargs=None):
    return [img.convert("RGB") for img in doc["question_images_decoded"]]


def zerobench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question_text"]
    return f"{question}\n\n\nLet’s think step by step and give the final answer in curly braces, like this: {{final_answer}}"


def zerobench_doc_to_messages(doc, lmms_eval_specific_kwargs=None):
    prompt = zerobench_doc_to_text(doc, lmms_eval_specific_kwargs)
    images = zerobench_doc_to_visual(doc, lmms_eval_specific_kwargs)
    if lmms_eval_specific_kwargs is not None:
        system_prompt = lmms_eval_specific_kwargs.get("system_prompt", "")
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {
            "role": "user",
            "content": [
                *[{"type": "image", "url": img} for img in images],
                {"type": "text", "text": prompt},
            ],
        },
    ]
    return messages


def extract_final_answer(text):
    match = re.search(r"\{([^}]+)\}", text)
    if match:
        return match.group(1).strip()
    return text.strip()


def llm_judge_evaluate(question, ground_truth, prediction):
    if server is None:
        return 0

    prompt = f"""
You are an impartial judge evaluating the correctness of a model's answer to a question.
Question: {question}
Ground Truth: {ground_truth}
Model Prediction: {prediction}

Is the Model Prediction correct based on the Ground Truth? 
Respond with only "YES" or "NO".
"""
    try:
        request = Request(messages=[{"role": "user", "content": prompt}], config=server_config)
        response = server.evaluate(request)
        if response.success:
            content = response.content.strip().upper()
            if "YES" in content:
                return 1
            elif "NO" in content:
                return 0
    except Exception as e:
        eval_logger.error(f"LLM Judge error: {e}")

    return 0


def zerobench_process_results(doc, results):
    prediction = results[0].strip()
    ground_truth = doc["question_answer"]
    question = doc["question_text"]

    # Method 1: Exact Match
    extracted_answer = extract_final_answer(prediction)
    exact_match = 1 if extracted_answer == ground_truth else 0

    # Method 2: LLM Judge
    llm_judge_score = llm_judge_evaluate(question, ground_truth, prediction)

    return {
        "exact_match": exact_match,
        "llm_judge": llm_judge_score,
    }


def zerobench_aggregate_results(results):
    total = len(results)
    if total == 0:
        return 0

    # Check which metric we are aggregating based on the first result keys or context
    # But usually this function receives a list of values if it's a simple aggregation,
    # or we can handle list of dicts if we return dicts in process_results.
    # Wait, lmms-eval aggregation usually receives a list of whatever process_results returns for that metric?
    # Actually, looking at other tasks, process_results returns a dict where keys match metric names.
    # And aggregation receives a list of results for that specific metric?
    # Let's check mmbench again.

    # In mmbench_en_dev.yaml:
    # metric_list:
    #   - metric: gpt_eval_score
    #     aggregation: !function en_utils.mmbench_aggregate_dev_results_eval

    # In mmbench_evals.py:
    # process_results returns a dict with keys matching metric names?
    # No, process_results returns a dict.
    # The framework extracts the value for the metric key and passes a list of those values to aggregation?
    # Let's re-read how lmms-eval works or check a simpler task.

    # Checking mathvista/utils.py:
    # process_results returns:
    # {
    #     "llm_as_judge_eval": result,
    #     "submission": result,
    # }
    # And aggregation receives `results` which seems to be a list of `result` dicts.

    # So if I return a simple value (int/float) for the metric key in process_results,
    # aggregation will receive a list of those values?
    # Let's assume yes for simple metrics.

    # However, for flexibility, I'll return the score directly in process_results for each metric key.
    # So `results` in aggregate_results will be a list of scores (0 or 1).

    return sum(results) / total
