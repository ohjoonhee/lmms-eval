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


def frozenlake_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def frozenlake_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    choices = [f"{chr(i + ord('A'))}. {choice}" for i, choice in enumerate(doc["choices"])]
    prompt = "\n".join([doc["question"]] + choices)

    if lmms_eval_specific_kwargs is not None:
        if "pre_prompt" in lmms_eval_specific_kwargs:
            prompt = lmms_eval_specific_kwargs["pre_prompt"] + "\n" + prompt
        if "post_prompt" in lmms_eval_specific_kwargs:
            prompt += "\n" + lmms_eval_specific_kwargs["post_prompt"]
    prompt += "\nAnswer with the option letter from the given choices directly."
    return prompt


def frozenlake_doc_to_messages(doc, lmms_eval_specific_kwargs=None):
    messages = []
    if lmms_eval_specific_kwargs is not None:
        if "system_prompt" in lmms_eval_specific_kwargs:
            messages.append({"role": "system", "content": [{"type": "text", "text": lmms_eval_specific_kwargs["system_prompt"]}]})

    user_content = []
    user_content.append({"type": "image", "url": frozenlake_doc_to_visual(doc)[0]})
    user_content.append({"type": "text", "text": frozenlake_doc_to_text(doc, lmms_eval_specific_kwargs)})
    messages.append({"role": "user", "content": user_content})
    return messages


def llm_judge_evaluate(question, ground_truth, prediction):
    if server is None:
        return 0

    prompt = f"""
You are an impartial judge evaluating the correctness of a model's answer to a multiple choice question.
Question and Options:
{question}

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


def frozenlake_process_results(doc, results):
    pred = results[0].strip()
    gt_index = doc["answer"]
    gt = chr(gt_index + ord("A"))

    full_question = frozenlake_doc_to_text(doc)

    score = llm_judge_evaluate(full_question, gt, pred)
    return {"accuracy": score}


def frozenlake_aggregate_results(results):
    if not results:
        return 0
    return sum(results) / len(results)
