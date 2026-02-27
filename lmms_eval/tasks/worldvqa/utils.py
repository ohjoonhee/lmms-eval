import base64
import os
from collections import defaultdict
from io import BytesIO

from loguru import logger as eval_logger
from PIL import Image

from lmms_eval.llm_judge import Request, ServerConfig, get_server

# Initialize LLM Judge
API_TYPE = os.getenv("API_TYPE", "openai")
MODEL_VERSION = os.getenv("MODEL_VERSION", "gpt-4o-2024-11-20")

try:
    server_config = ServerConfig(
        model_name=MODEL_VERSION, temperature=0.0, max_tokens=1024
    )
    server = get_server(server_name=API_TYPE, config=server_config)
except Exception as e:
    eval_logger.warning(
        f"Failed to initialize LLM Judge: {e}. LLM Judge metric will fail if used."
    )
    server = None

# Official WorldVQA judge prompt from:
# https://github.com/MoonshotAI/WorldVQA/blob/master/eval/eval.py
JUDGE_WORLDQA_PROMPT = """
### Role
You are an expert judge specialized in evaluating the correctness of answers. Your task is to assess whether a model-generated answer is correct based on a given question, the model's response, and the ground truth answer.

### Task: Evaluate Answer Correctness
Please classify the model's response into one of the following three categories. Ignore differences in formatting, punctuation, language (Chinese vs. English), or abbreviations/full names. Focus strictly on the **core semantics** and the **level of detail (granularity)**:

1. **Correct**:
    - The model answer contains the core information of the ground truth.
    - The model answer is semantically consistent with the ground truth and contains no contradictions.
    - **The granularity of the model answer is equal to or finer than the ground truth.**
    - Extra irrelevant information is allowed as long as it does not conflict with the ground truth.

2. **Incorrect**:
    - The model answer provides information that contradicts the ground truth.
    - The model answer provides the wrong specific entity, value, or description.
    - **The granularity of the model answer is coarser than the ground truth**, leading to incomplete or insufficiently specific information.
    - Even if the model expresses uncertainty but follows up with a wrong answer (e.g., "I'm not sure, maybe it's B" when the truth is A), it is considered Incorrect.

3. **Unattempted**:
    - The model explicitly states it does not know the answer (e.g., "I don't know," "I cannot answer this question").
    - The model suggests the user search elsewhere (e.g., "Please search the internet").
    - The model answer contains no information from the ground truth but provides no incorrect or contradictory information.

### Output Format
Please strictly follow this two-line format for your output:
1. **Evaluation**: [A brief explanation of your reasoning]
2. **Label**: [Final classification: "Correct", "Incorrect", or "Unattempted"]

---
### Examples

**Example 1 (Incorrect - Granularity Mismatch/Too Coarse)**
Input:
'''
Question: 图片中属于什么类型的田地？
Model Answer: 图片中展示的是梯田。梯田是在山坡地上开垦并修筑的阶梯状农田。
Ground Truth Answer: 龙脊梯田
'''
Evaluation: 标准答案特指"龙脊梯田"，模型只回答了通用的"梯田"。模型答案层级比答案层级更粗略，未能提供标准答案所需的特指信息，属于层级不一致导致的回答错误。
Label: Incorrect

**Example 2 (Correct - Finer Granularity)**
Input:
'''
Question: What weather phenomenon is in the image?
Model Answer: Based on the visual evidence in the image, the weather phenomenon shown is a **severe storm with extremely high winds**, most likely a **tornado** or a very powerful **hurricane/typhoon**.
Ground Truth Answer: High winds
'''
Evaluation: The ground truth is "high winds," and a "tornado" is a more specific and granular type of high wind. The semantics are correct and the detail is finer.
Label: Correct

**Example 3 (Correct)**
Input:
'''
Question: 图中内容是什么品牌的logo？
Model Answer: via浏览器
Ground Truth Answer: via
'''
Evaluation: 模型答案"via浏览器"包含了标准答案"via"，核心语义一致，且"via浏览器"是更具体的描述，层级上是匹配的。
Label: Correct

**Example 4 (Unattempted)**
Input:
'''
Question: Which athlete is in the image?
Model Answer: I cannot answer this question as I do not have relevant sports data.
Ground Truth Answer: Wout Weghorst
'''
Evaluation: The model explicitly states its inability to answer and provides no incorrect information.
Label: Unattempted

**Example 5 (Incorrect)**
Input:
'''
Question: 图片中展示的是什么苹果品种？
Model Answer: 我觉得可能是阿克苏苹果。
Ground Truth Answer: 烟台苹果
'''
Evaluation: 虽然模型用了"可能"等词汇，但它给出的具体答案"阿克苏苹果"与标准答案"烟台苹果"不符，提供了错误信息。
Label: Incorrect

**Example 6 (Unattempted)**
Input:
'''
Question: What is the name of the insect in this image?
Model Answer: This is a photo of an insect. To find the species, consult an entomologist or use recognition software.
Ground Truth Answer: Japanese rhinoceros beetle
'''
Evaluation: The model does not attempt to name the insect and suggests the user search elsewhere, providing no incorrect information.
Label: Unattempted

---
### Current Task
Input:
'''
Question: {question}
Model Answer: {model_answer}
Ground Truth Answer: {ground_truth_answer}
'''

Evaluation:
"""


def _decode_base64_image(b64_string: str) -> Image.Image:
    """Decode a base64-encoded PNG string to a PIL Image."""
    image_bytes = base64.b64decode(b64_string)
    return Image.open(BytesIO(image_bytes)).convert("RGB")


def worldvqa_doc_to_visual(doc):
    """Extract image from document. Images are base64-encoded PNG strings."""
    return [_decode_base64_image(doc["image"])]


def worldvqa_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """Format the prompt with language-aware prefix and model-specific pre/post prompts."""
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")

    question = doc["question"]

    # Use language-appropriate detail prompt if no custom pre_prompt is set
    if not pre_prompt:
        language = doc.get("language", "en")
        if language == "zh":
            pre_prompt = "请尽可能提供详细的回答。\n"
        else:
            pre_prompt = "Please provide as much detail as possible in your answer. \n"

    return f"{pre_prompt}{question}{post_prompt}"


def worldvqa_doc_to_messages(doc, lmms_eval_specific_kwargs=None):
    """Convert document to messages for chat-based models."""
    imgs = worldvqa_doc_to_visual(doc)
    text = worldvqa_doc_to_text(doc, lmms_eval_specific_kwargs)

    messages = []
    if lmms_eval_specific_kwargs:
        system_prompt = lmms_eval_specific_kwargs.get(
            "system_instruction"
        ) or lmms_eval_specific_kwargs.get("system_prompt")
        if system_prompt:
            messages.append(
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
            )

    user_content = []
    for img in imgs:
        user_content.append({"type": "image", "url": img})
    user_content.append({"type": "text", "text": text})

    messages.append({"role": "user", "content": user_content})

    return messages


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from model output.

    Matches the official WorldVQA eval logic:
    - If both <think> and </think> present, take content after </think>
    - If only 'think>' present (malformed), take content after last 'think>'
    """
    if "<think>" in text and "</think>" in text:
        return text.split("</think>")[-1].strip()
    elif "think>" in text:
        return text.split("think>")[-1].strip()
    return text.strip()


def _judge_evaluate(question: str, ground_truth: str, prediction: str) -> str:
    """
    Judge a prediction using the official WorldVQA 3-way judge prompt.

    Returns:
        One of "correct", "incorrect", or "unattempted".
    """
    if server is None:
        eval_logger.warning(
            "LLM Judge not initialized. Returning 'incorrect' by default."
        )
        return "incorrect"

    prompt = JUDGE_WORLDQA_PROMPT.format(
        question=question,
        ground_truth_answer=ground_truth,
        model_answer=prediction,
    )

    try:
        request = Request(
            messages=[{"role": "user", "content": prompt}], config=server_config
        )
        response = server.evaluate(request)
        if response.success:
            content = response.content
            if "Correct" in content:
                return "correct"
            elif "Unattempted" in content:
                return "unattempted"
            else:
                return "incorrect"
    except Exception as e:
        eval_logger.error(f"LLM Judge error: {e}")

    return "incorrect"


def worldvqa_process_results(doc, results):
    """
    Process model results using LLM judge evaluation.

    Args:
        doc: Dataset document with question, answer, category, language, difficulty.
        results: List with single element (model's generated text).

    Returns:
        Dict with metric names as keys.
    """
    pred = results[0] if results else ""
    pred = _strip_thinking(pred)

    question = doc["question"]
    ground_truth = doc["answer"]
    category = doc.get("category", "unknown")
    language = doc.get("language", "unknown")
    difficulty = doc.get("difficulty", "unknown")

    # LLM judge evaluation (3-way: correct/incorrect/unattempted)
    judge_label = _judge_evaluate(question, ground_truth, pred)
    judge_score = 1.0 if judge_label == "correct" else 0.0

    result_entry = {
        "category": category,
        "language": language,
        "difficulty": difficulty,
        "score": judge_score,
        "judge_label": judge_label,
        "answer_category": judge_label,
        "prediction": pred,
        "ground_truth": ground_truth,
    }

    return {
        "worldvqa_accuracy": result_entry,
    }


def worldvqa_aggregate_results(results: list[dict]) -> float:
    """
    Aggregate WorldVQA results and log per-category, per-difficulty, per-language breakdowns.

    Args:
        results: List of result dicts from process_results.

    Returns:
        Overall accuracy as a percentage.
    """
    if not results:
        return 0.0

    all_scores = []
    category_scores: dict[str, list[float]] = defaultdict(list)
    difficulty_scores: dict[str, list[float]] = defaultdict(list)
    language_scores: dict[str, list[float]] = defaultdict(list)
    answer_category_counts: dict[str, int] = defaultdict(int)

    for r in results:
        score = r["score"]
        all_scores.append(score)
        category_scores[r["category"]].append(score)
        difficulty_scores[r["difficulty"]].append(score)
        language_scores[r["language"]].append(score)
        answer_category_counts[r["answer_category"]] += 1

    # Log difficulty breakdown (matches official eval order)
    eval_logger.info("WorldVQA Results by Difficulty:")
    for diff in ["easy", "medium", "hard"]:
        scores = difficulty_scores.get(diff, [])
        if scores:
            acc = sum(scores) / len(scores) * 100.0
            eval_logger.info(
                f"  {diff.capitalize()}: {acc:.2f}% = {int(sum(scores))}/{len(scores)}"
            )

    # Log category breakdown
    eval_logger.info("WorldVQA Results by Category:")
    for cat, scores in sorted(category_scores.items()):
        acc = sum(scores) / len(scores) * 100.0
        eval_logger.info(f"  {cat}: {acc:.2f}% = {int(sum(scores))}/{len(scores)}")

    # Log language breakdown
    eval_logger.info("WorldVQA Results by Language:")
    for lang, scores in sorted(language_scores.items()):
        acc = sum(scores) / len(scores) * 100.0
        eval_logger.info(f"  {lang}: {acc:.2f}% = {int(sum(scores))}/{len(scores)}")

    # Log answer category breakdown (correct/incorrect/unattempted)
    total = len(results)
    eval_logger.info("WorldVQA Answer Category Breakdown:")
    for cat in ["correct", "incorrect", "unattempted"]:
        count = answer_category_counts.get(cat, 0)
        rate = count / total * 100.0 if total > 0 else 0.0
        eval_logger.info(f"  {cat}: {count}/{total} ({rate:.2f}%)")

    # Overall accuracy
    overall_acc = sum(all_scores) / len(all_scores) * 100.0
    eval_logger.info(
        f"WorldVQA Overall: {overall_acc:.2f}% = {int(sum(all_scores))}/{len(all_scores)}"
    )

    return overall_acc
