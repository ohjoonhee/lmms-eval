import os
import re

from loguru import logger as eval_logger

from lmms_eval.filters.extraction import ExtendedRegexFilter
from lmms_eval.filters.transformation import MapFilter
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

    server = None


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


REPLACE_PROMPT = "Please answer directly with only the letter of the correct option and nothing else."


def realworldqa_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def realworldqa_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    pre_prompt = ""
    post_prompt = ""
    question = doc["question"].strip()
    if "pre_prompt" in lmms_eval_specific_kwargs:
        pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    if "post_prompt" in lmms_eval_specific_kwargs and lmms_eval_specific_kwargs["post_prompt"]:
        question = question.replace(REPLACE_PROMPT, "")
        post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    return f"{pre_prompt}{question}{post_prompt}"


# number_words_to_digits = {
#     "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
#     "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
#     "ten": "10"
# }


def realworldqa_process_results(doc, results):
    raw_pred = results[0]
    # Strip thinking process
    pred_stripped = raw_pred.split("</think>")[-1].strip()

    # Exact Match logic
    pred_em = pred_stripped.lower().strip().rstrip(".")
    gt_ans = doc["answer"].lower().strip()

    score = 1.0 if pred_em == gt_ans else 0.0

    # LLM Judge logic
    question_text = realworldqa_doc_to_text(doc, {})
    judge_score = llm_judge_evaluate(question_text, doc["answer"], pred_stripped)

    return {
        "exact_match": score,
        "llm_judge": judge_score,
    }


class NumberWordsToDigitsFilter(MapFilter):
    def __init__(self) -> None:
        mapping_dict = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"}
        super().__init__(mapping_dict, default_value=None)

    def apply(self, resps, docs):
        def filter_set(inst):
            return [self.mapping_dict.get(resp.lower(), resp) for resp in inst]

        return [filter_set(resp) for resp in resps]


class MultiChoiceRegexFilter(ExtendedRegexFilter):
    def __init__(self, *args, **kwargs):
        """
        regex_pattern: The basic regex pattern to use. If fails to match, we will use the customized match procedure
                        - step 1 : We parse the choices between ([A-Z])s then try to find these choices in the response.
                        - step 2 : We parse the choice with regex :[\s]*([A-?]), where ? varies by number of choices.
        group_select: Selects the (group_select)th match from the findall result.
        ignore_case: Ignores the case during step 1 matching
        ignore_punctuation: Remove the punctuation during step 1 matching
        regexes_to_ignore: Remove these regexes during step 1 matching
        """
        super().__init__(*args, **kwargs)

    def apply(self, resps, docs):
        # here, we assume we have a list, in which each element is
        # a list of model responses for some particular input/target pair.
        # so we process each of these (same input/target response sets)
        # independently (and keep them a list.)

        filtered_resps = []

        for r, doc in zip(resps, docs):
            fallback_regexes = []
            choice_to_alpha = {}
            next_alpha = "A"

            without_paren_fallback_regexes = []
            without_paren_to_target = {}

            # Regex to extract multiple choice options from the question
            multiple_choices_regex = re.compile(r"\b([A-Z])\.\s+([^\n]*)")
            matches = multiple_choices_regex.findall(doc["question"])

            # Build regex patterns and mappings for each choice
            for m in matches:
                choice_text = m[1].strip()
                fallback_regexes.append(f"{re.escape(choice_text)}")
                choice_to_alpha[choice_text] = next_alpha

                next_alpha = chr(ord(next_alpha) + 1)

            # Compile regex to match any of the extracted choices
            fallback_regex = re.compile("|".join(fallback_regexes))

            # Process each response
            filtered = []
            for resp in r:
                # Remove any punctuation and extra spaces
                cleaned_resp = re.sub(r"[^\w\s]", "", resp).strip()
                # Try to match cleaned response with the choice text
                match = fallback_regex.search(cleaned_resp)
                if match and match.group() in choice_to_alpha:
                    # Map the matched choice text back to its corresponding letter
                    filtered.append(choice_to_alpha[match.group()])
                else:
                    # If no match, return the cleaned response
                    filtered.append(cleaned_resp)

            filtered_resps.append(filtered[0])

        return filtered_resps
