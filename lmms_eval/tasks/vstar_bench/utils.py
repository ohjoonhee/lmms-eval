import re
import os
from collections import defaultdict

from loguru import logger as eval_logger

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file
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


def vstar_doc_to_visual(doc):
    """Convert document to visual input."""
    return [doc["image"].convert("RGB")]


def vstar_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """Convert document to text prompt with options."""
    question = doc["question"]

    # Handle choices
    choices = doc["choices"]

    choices = [f"{chr(i + ord('A'))}. {choice}" for i, choice in enumerate(choices)]
    text = "\n".join([question] + choices)

    # Add pre-prompt and post-prompt if specified
    if lmms_eval_specific_kwargs:
        if "pre_prompt" in lmms_eval_specific_kwargs and lmms_eval_specific_kwargs["pre_prompt"]:
            text = f"{lmms_eval_specific_kwargs['pre_prompt']}{text}"
        if "post_prompt" in lmms_eval_specific_kwargs and lmms_eval_specific_kwargs["post_prompt"]:
            text = f"{text}{lmms_eval_specific_kwargs['post_prompt']}"

    return text


def vstar_doc_to_messages(doc, lmms_eval_specific_kwargs=None):
    """Convert document to messages for chat-based models."""
    imgs = vstar_doc_to_visual(doc)
    text = vstar_doc_to_text(doc, lmms_eval_specific_kwargs)

    messages = []
    if "system_prompt" in lmms_eval_specific_kwargs and lmms_eval_specific_kwargs["system_prompt"]:
        system_prompt = lmms_eval_specific_kwargs["system_prompt"]
        messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})

    user_content = []
    for img in imgs:
        user_content.append({"type": "image", "url": img})
    user_content.append({"type": "text", "text": text})

    messages.append({"role": "user", "content": user_content})

    return messages


def extract_answer_letter(response):
    """Extract the answer letter from model response."""
    # Clean the response
    response = response.strip().upper()

    # Clean the <think> tags if present
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

    # Try to find patterns like "A", "(A)", "A.", "A)", "Answer: A", etc.
    patterns = [
        r"^([A-D])\s*[\.)\]]*",  # Starts with letter
        r"(?:THE\s+)?(?:ANSWER|CHOICE|OPTION)(?:\s+IS)?[\s:]+([A-D])",  # Answer: A or The answer is A format
        r"\(([A-D])\)",  # (A) format
        r"([A-D])\s*(?:\.|\)|])",  # A. or A) or A] format
        r"(?:^|\s)([A-D])(?:\s|$)",  # Standalone letter
    ]

    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(1).upper()

    # If no pattern matches, check if response contains only one letter A-D
    letters = re.findall(r"[A-D]", response)
    if len(letters) == 1:
        return letters[0]

    # Default to first character if it's A-D
    if response and response[0] in "ABCD":
        return response[0]

    return None


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


def vstar_process_results(doc, results):
    """
    Process the model results and compare with ground truth.

    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name, value: metric value
    """
    pred = results[0] if results else ""

    # Extract predicted answer letter
    pred_letter = extract_answer_letter(pred)

    # The label in the dataset should be the correct answer letter (A, B, C, or D)
    gt_letter = ["A", "B", "C", "D"][doc["answer"]]

    # Score 1 if correct, 0 otherwise
    score = 1.0 if pred_letter == gt_letter else 0.0

    # Get category for aggregation
    category = doc.get("category", "unknown")

    # Log for debugging
    if score == 0:
        eval_logger.debug(f"Question: {doc['question'][:100]}...")
        eval_logger.debug(f"Model response: {pred}")
        eval_logger.debug(f"Predicted: {pred_letter}, Ground Truth: {gt_letter}")
        eval_logger.debug(f"Raw prediction: {pred}")

    # Calculate LLM Judge score
    text = vstar_doc_to_text(doc)
    judge_score = llm_judge_evaluate(text, gt_letter, pred)

    # Return metrics for different aggregations
    result_acc = {"question_id": doc["question_id"], "category": category, "score": score, "prediction": pred_letter, "ground_truth": gt_letter}
    result_judge = {"question_id": doc["question_id"], "category": category, "score": judge_score, "prediction": pred_letter, "ground_truth": gt_letter}

    return {
        f"vstar_{category}_acc": result_acc,
        "vstar_overall_acc": result_acc,
        f"vstar_{category}_llm_judge": result_judge,
        "vstar_overall_llm_judge": result_judge,
    }


def vstar_aggregate_results(results):
    """
    Aggregate results by category and overall.

    Args:
        results: a list of values returned by process_results
    Returns:
        Aggregated accuracy score
    """
    if not results:
        return 0.0

    # Group by category
    category_scores = defaultdict(list)
    all_scores = []

    for result in results:
        category = result["category"]
        score = result["score"]
        category_scores[category].append(score)
        all_scores.append(score)

    # Calculate accuracy for each category
    for category, scores in category_scores.items():
        if scores:
            acc = sum(scores) / len(scores) * 100.0
            eval_logger.info(f"{category}: {acc:.2f}% (n={len(scores)})")

    # Calculate overall accuracy
    if all_scores:
        overall_acc = sum(all_scores) / len(all_scores) * 100.0
        eval_logger.info(f"Overall: {overall_acc:.2f}% (n={len(all_scores)})")
        return overall_acc

    return 0.0
