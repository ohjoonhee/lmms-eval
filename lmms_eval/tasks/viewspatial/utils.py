import re


def viewspatial_doc_to_text(doc, lmms_eval_specific_kwargs: Optional[dict] = None):
    """Extracts the text prompt from a viewspatial dataset sample.

    Args:
        doc (dict): A viewspatial dataset sample dictionary.
        lmms_eval_specific_kwargs (dict, optional): Model-specific kwargs with pre_prompt/post_prompt.

    Returns:
        str: The input prompt text.
    """
    question = doc["question"]
    choices = doc["choices"]

    # prompt format from viewspatial bench paper
    question_text = f"Question: {question}\n"
    choices_text = f"Choices: {choices}\n"

    pre_prompt = ""
    post_prompt = "Reply only to the corresponding option.\nAnswer:"
    if lmms_eval_specific_kwargs:
        if "pre_prompt" in lmms_eval_specific_kwargs and lmms_eval_specific_kwargs["pre_prompt"]:
            pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
        if "post_prompt" in lmms_eval_specific_kwargs and lmms_eval_specific_kwargs["post_prompt"]:
            post_prompt = lmms_eval_specific_kwargs["post_prompt"]

    prompt = question_text + choices_text + post_prompt
    if pre_prompt:
        prompt = f"{pre_prompt}\n{prompt}"
    return prompt


def viewspatial_doc_to_messages(doc, lmms_eval_specific_kwargs: Optional[dict] = None):
    """Convert document to messages for chat-based models.

    Args:
        doc (dict): A viewspatial dataset sample dictionary.
        lmms_eval_specific_kwargs (dict, optional): Model-specific kwargs with pre_prompt/post_prompt.

    Returns:
        list[dict]: Messages in the chat protocol format.
    """
    imgs = viewspatial_doc_to_visual(doc)
    text = viewspatial_doc_to_text(doc, lmms_eval_specific_kwargs)

    messages = []
    if lmms_eval_specific_kwargs:
        system_prompt = lmms_eval_specific_kwargs.get("system_instruction") or lmms_eval_specific_kwargs.get("system_prompt")
        if system_prompt:
            messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})

    user_content = []
    for img in imgs:
        user_content.append({"type": "image", "url": img})
    user_content.append({"type": "text", "text": text})

    messages.append({"role": "user", "content": user_content})

    return messages


def viewspatial_doc_to_visual(doc):
    """Extracts and converts visual data from a viewspatial dataset sample.

    This function iterates over the 'images' key in the dataset sample, calling
    the .convert("RGB") method on each item (presumed to be a PIL/Pillow Image).

    Args:
        doc (dict): A viewspatial dataset sample dictionary.

    Returns:
        list: A list of visual elements (e.g., PIL Images) converted to 'RGB' format.
    """
    return [visual.convert("RGB") for visual in doc["images"]]


def extract_option(text):
    match = re.search(r"\b([A-D])\b", text, re.IGNORECASE)
    return match.group(1).upper() if match else None


def viewspatial_process_results(doc, results):
    """Processes the model's output for a single viewspatial document."""
    # extract grounded answer
    grounded_output = doc["answer"]
    grounded_option = extract_option(grounded_output)
    # eval_logger.info(f"Grounded answer: {grounded_output}")

    # extract predicted answer
    pred = results[0]
    pred = pred.split("\n")[-1]
    pred_answer = extract_option(pred)
    # eval_logger.info(f"Predicted answer: {pred_answer}")

    score = 1.0 if pred_answer == grounded_option else 0.0
    # eval_logger.info(f"Score: {score}")

    return {"overall_accuracy": {"score": score}}


def viewspatial_aggregate_results(results):
    """Aggregates the 'overall_accuracy' results.

    Calculates the mean score from a list of result dictionaries.

    Args:
        results (list[dict]): A list of dictionaries, where each dict has
                              a 'score' key (e.g., [{'score': 1.0}, ...]).

    Returns:
        float: The average score (mean accuracy).
    """
    # --- Compute the total score across all results ---
    total_score = 0.0
    for res in results:
        total_score += res["score"]

    # --- Compute average score safely ---
    avg_score = total_score / len(results) if results else 0.0
    return avg_score
