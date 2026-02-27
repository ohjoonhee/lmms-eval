import json
import os
from collections import defaultdict
from typing import Optional, Tuple

from loguru import logger as eval_logger

from lmms_eval.llm_judge import Request, ServerConfig, get_server
from lmms_eval.llm_judge.base import ServerInterface

# Lazy-initialized LLM Judge for evaluation.
# The server and config are created on first use (not at import time) to avoid
# creating HTTP clients / connection pools that persist for the entire process
# lifetime even when judging is never invoked.
_server: Optional[ServerInterface] = None
_server_config: Optional[ServerConfig] = None


def _get_judge() -> Tuple[ServerInterface, ServerConfig]:
    """Return the (server, config) pair, creating them on first call."""
    global _server, _server_config
    if _server is None:
        api_type = os.getenv("API_TYPE", "openai")
        model = os.getenv("MODEL_VERSION", "gpt-4o-2024-11-20")
        _server_config = ServerConfig(
            model_name=model, temperature=0.0, max_tokens=1024
        )
        _server = get_server(server_name=api_type, config=_server_config)
    return _server, _server_config


# ---------------------------------------------------------------------------
# Official Pix2Fact prompt template (from pix2fact_eval/src/prompt.py)
# ---------------------------------------------------------------------------
PROMPT_TEMPLATE = """You are a highly specialized AI designed to function as an automated visual analysis API. Your sole function is to analyze an image and a question provided by the user, and return your entire response as a single, valid JSON object.
--- RULES ---
Your entire output MUST be a single, valid JSON object.
Your response MUST start with { and end with }.
DO NOT output ANY text, explanations, apologies, or markdown formatting (like ```json) before or after the JSON object. Your response must be the raw JSON and nothing else.
The JSON object MUST contain these exact five key: "Observation", "Search Plan", "Search Query", "Comprehensive Answer", and "Final Answer". Adhere strictly to this schema.
Limit your reasoning token under 4000. Do not use function call to address this task.
--- KEY DEFINITIONS & SCHEMA ---
"Observation": (String) Describe specific visual details from the image URL relevant to the question.
"Search Plan": (List of Strings) Outline a step-by-step plan to find the necessary information online.
"Search Query": (List of Strings) Extract the exact search queries from your Search Plan.
"Comprehensive Answer": (String) Provide a comprehensive, final answer integrating observations and search results.
"Final Answer": (String) Provide only the core, direct answer. If a definitive, factual answer (e.g., a specific name, date, number) cannot be determined, you MUST output the exact string '[NO_DEFINITIVE_ANSWER]' in this field.
--- ONE-SHOT EXAMPLE ---
This is an example of a user request and your expected output.
User Input Example:
Input Question: Who was the president of the USA when the book with 'Kjell' on its cover in the picture published?
Input Image: <image data here>

Your Expected JSON Output Example:
{
    "Observation": "On the top shelf of the book cart in the foreground, facing left, a book with a dark cover is visible. The author's name, \\"Kjell Westö,\\" is printed in white, and below it is the title, \\"Hagring 38.\\"",
    "Search Plan": [
        "Find the original publication date of the book titled \\"Hagring 38\\" by Kjell Westö.",
        "Identify who was the President of the United States during the publication year of the book."
    ],
    "Search Query": [
        "Hagring 38 Kjell Westö publication date",
        "who was US president in 2013"
    ],
    "Comprehensive Answer": "The book visible in the image is \\"Hagring 38\\" by Kjell Westö, which was originally published in 2013. In that year, the president of the USA was Barack Obama, who was in his second term.",
    "Final Answer": "Barack Obama"
}
--- YOUR TASK ---"""


# ---------------------------------------------------------------------------
# Official Pix2Fact judge prompt template (from pix2fact_eval/src/judge.py)
# ---------------------------------------------------------------------------
JUDGE_PROMPT_TEMPLATE = """You are a strict judge. Compare a Ground Truth answer vs a Model answer.

Rules:
1) Output ONLY one token: True or False (case-sensitive, no punctuation, no space, no code fences).
2) True if and only if the Model answer semantically matches the Ground Truth with respect to meaning and exact factual content.
   - Numbers/dates/names must match.
   - Language or casing differences are acceptable if meaning is identical.
   - If Ground Truth is '[NO_DEFINITIVE_ANSWER]', output True only if Model answer is exactly '[NO_DEFINITIVE_ANSWER]'.
3) If uncertain for any reason, output False.

Ground Truth: {ground_truth}
Model Answer: {model_answer}
"""


# ---------------------------------------------------------------------------
# Document conversion functions
# ---------------------------------------------------------------------------
def pix2fact_doc_to_visual(doc: dict) -> list:
    """Extract the PIL image from the dataset document."""
    return [doc["image"].convert("RGB")]


def pix2fact_doc_to_text(
    doc: dict, lmms_eval_specific_kwargs: dict | None = None
) -> str:
    """Build the text prompt for the model.

    By default uses the official Pix2Fact prompt template. When *pre_prompt*
    or *post_prompt* are set via ``lmms_eval_specific_kwargs``, a simpler
    ``{pre_prompt}{question}{post_prompt}`` format is used instead.
    """
    question = doc["question"]

    pre_prompt = ""
    post_prompt = ""
    if lmms_eval_specific_kwargs:
        pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
        post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")

    if pre_prompt or post_prompt:
        return f"{pre_prompt}{question}{post_prompt}"

    # Default: official Pix2Fact structured prompt
    return f"{PROMPT_TEMPLATE}\nInput Question: {question}\nInput Image: "


def pix2fact_doc_to_messages(
    doc: dict, lmms_eval_specific_kwargs: dict | None = None
) -> list:
    """Build chat messages with interleaved image and text for chat models."""
    imgs = pix2fact_doc_to_visual(doc)
    text = pix2fact_doc_to_text(doc, lmms_eval_specific_kwargs)

    messages: list[dict] = []

    # Optional system instruction
    if lmms_eval_specific_kwargs:
        system_prompt = lmms_eval_specific_kwargs.get(
            "system_instruction"
        ) or lmms_eval_specific_kwargs.get("system_prompt")
        if system_prompt:
            messages.append(
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
            )

    user_content: list[dict] = []
    for img in imgs:
        user_content.append({"type": "image", "url": img})
    user_content.append({"type": "text", "text": text})

    messages.append({"role": "user", "content": user_content})
    return messages


# ---------------------------------------------------------------------------
# Answer extraction (mirrors official pix2fact_eval/src/judge.py)
# ---------------------------------------------------------------------------
def _normalize_json(json_str: str) -> str:
    """Strip markdown fences and isolate the outermost JSON object."""
    if "```json" in json_str:
        json_str = json_str.replace("```json", "")
    if "```" in json_str:
        json_str = json_str.replace("```", "")

    idx_open = json_str.rfind("{")
    if idx_open != -1:
        json_str = json_str[idx_open:].strip()
    idx_close = json_str.rfind("}")
    if idx_close != -1:
        json_str = json_str[: idx_close + 1].strip()
    return json_str


def extract_final_answer(model_output: str) -> str:
    """Extract the ``Final Answer`` value from the model's JSON output.

    Follows the official Pix2Fact extraction logic:
    1. Parse JSON and look for ``"Final Answer"`` key.
    2. Fall back to alternative keys (``"Final"``, ``"answer"``, etc.).
    3. If JSON parsing fails entirely, return the raw (stripped) output so
       the LLM judge can still attempt a comparison.
    """
    if not isinstance(model_output, str) or not model_output.strip():
        return "Failed to answer"

    # Strip thinking / reasoning tags
    if "</think>" in model_output:
        model_output = model_output.split("</think>")[-1].strip()

    try:
        parsed = json.loads(_normalize_json(model_output))
        if isinstance(parsed, dict):
            if "Final Answer" in parsed:
                return str(parsed["Final Answer"])
            for key in ("Final", "answer", "final_answer", "final answer"):
                if key in parsed:
                    return str(parsed[key])
        return "Failed to answer"
    except Exception:
        # JSON parsing failed – return trimmed raw output as a best-effort
        # answer so the LLM judge can still attempt a semantic comparison.
        raw = model_output.strip()
        return raw if raw else "Failed to answer"


# ---------------------------------------------------------------------------
# LLM Judge evaluation
# ---------------------------------------------------------------------------
def _judge_answer(ground_truth: str, model_answer: str) -> int:
    """Compare *model_answer* against *ground_truth* using the LLM judge.

    Returns 1 for correct (``True``), 0 for incorrect (``False`` or error).
    Uses the official Pix2Fact judge prompt which outputs ``True``/``False``.
    """
    try:
        server, config = _get_judge()
    except Exception as e:
        eval_logger.warning(f"LLM Judge not available: {e}. Returning score 0.")
        return 0

    prompt = JUDGE_PROMPT_TEMPLATE.format(
        ground_truth=ground_truth,
        model_answer=model_answer,
    )

    try:
        request = Request(
            messages=[{"role": "user", "content": prompt}],
            config=config,
        )
        response = server.evaluate(request)
        if response.success:
            content = response.content.strip()
            if content == "True":
                return 1
            if content == "False":
                return 0
            # Lenient fallback: check for substring
            if "True" in content:
                return 1
    except Exception as e:
        eval_logger.error(f"LLM Judge error: {e}")

    return 0


# ---------------------------------------------------------------------------
# Result processing and aggregation
# ---------------------------------------------------------------------------
def pix2fact_process_results(doc: dict, results: list) -> dict:
    """Process a single document's model output through the LLM judge.

    Returns a dict keyed by metric name mapping to the result payload.
    """
    pred = results[0] if results else ""
    ground_truth = str(doc["answer"])

    final_answer = extract_final_answer(pred)
    score = _judge_answer(ground_truth, final_answer)

    return {
        "pix2fact_accuracy": {
            "doc_id": doc.get("item_id", ""),
            "score": score,
            "image_category": doc.get("image_category", "unknown"),
            "visual_perception_type": doc.get("visual_perception_type", "unknown"),
            "knowledge_domain": doc.get("knowledge_domain", "unknown"),
            "reasoning_logic_type": doc.get("reasoning_logic_type", "unknown"),
        },
    }


def pix2fact_aggregate_results(results: list) -> float:
    """Compute overall accuracy and log per-category breakdowns.

    Returns accuracy as a percentage (0–100).
    """
    if not results:
        return 0.0

    scores = [r["score"] for r in results]
    total = len(scores)
    correct = sum(scores)

    # Per-category breakdowns
    category_keys = [
        ("image_category", "Image Category"),
        ("visual_perception_type", "Visual Perception Type"),
        ("knowledge_domain", "Knowledge Domain"),
        ("reasoning_logic_type", "Reasoning Logic Type"),
    ]
    for key, label in category_keys:
        grouped: dict[str, list[int]] = defaultdict(list)
        for r in results:
            grouped[r[key]].append(r["score"])

        eval_logger.info(f"\n--- {label} breakdown ---")
        for cat in sorted(grouped):
            cat_scores = grouped[cat]
            acc = sum(cat_scores) / len(cat_scores) * 100.0
            eval_logger.info(f"  {cat}: {acc:.2f}% (n={len(cat_scores)})")

    overall = correct / total * 100.0
    eval_logger.info(f"\nOverall Pix2Fact accuracy: {overall:.2f}% ({correct}/{total})")
    return overall
