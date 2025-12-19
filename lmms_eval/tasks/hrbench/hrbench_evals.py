import copy as cp
import os
import string
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from loguru import logger as eval_logger
from tqdm import tqdm

from lmms_eval.llm_judge import Request, ServerConfig, get_server


class HRBenchEval:
    def __init__(self, api_key, gpt_model="gpt-3.5-turbo", max_workers=12):
        self.api_key = api_key
        self.gpt_model = gpt_model
        self.max_workers = max_workers

        # Initialize the judge server
        api_type = os.getenv("API_TYPE", "openai")
        self.server_config = ServerConfig(model_name=self.gpt_model, temperature=0, max_tokens=256)
        self.server = get_server(server_name=api_type, config=self.server_config)

    def can_infer_option(self, answer, choices):
        verbose = os.environ.get("VERBOSE", 0)
        # Choices is a dictionary
        if "Failed to obtain answer via API" in answer:
            return False

        reject_to_answer = ["Sorry, I can't help with images of people yet.", "I can't process this file.", "I'm sorry, but without the image provided", "Cannot determine the answer"]
        for err in reject_to_answer:
            if err in answer:
                return "Z"

        def count_choice(splits, choices, prefix="", suffix=""):
            cnt = 0
            for c in choices:
                if prefix + c + suffix in splits:
                    cnt += 1
            return cnt

        answer_mod = cp.copy(answer)
        chars = ".()[],:;!*#{}"
        for c in chars:
            answer_mod = answer_mod.replace(c, " ")

        splits = [x.strip() for x in answer_mod.split()]
        count = count_choice(splits, choices)

        if count == 1:
            for ch in choices:
                if "A" in splits and len(splits) > 3 and verbose:
                    return False
                if ch in splits:
                    return ch
        elif count == 0 and count_choice(splits, {"Z", ""}) == 1:
            return "Z"
        return False

    def can_infer_text(self, answer, choices):
        answer = answer.lower()
        assert isinstance(choices, dict)
        for k in choices:
            assert k in string.ascii_uppercase
            choices[k] = str(choices[k]).lower()
        cands = []
        for k in choices:
            if choices[k] in answer:
                cands.append(k)
        if len(cands) == 1:
            return cands[0]
        return False

    def can_infer(self, answer, choices):
        answer = str(answer)
        copt = self.can_infer_option(answer, choices)
        return copt if copt else self.can_infer_text(answer, choices)

    def get_chat_response(self, data, temperature=0, max_tokens=256, patience=10, sleep_time=0):
        question = data["question"]
        options = data["options"]
        prediction = data["prediction"]
        ret = self.can_infer(prediction, options)
        if ret:
            data["gpt_prediction"] = ret
            return data

        prompt = self.build_prompt(question, options, prediction)
        messages = [
            {"role": "user", "content": prompt},
        ]
        
        # Create config for this specific request
        config = ServerConfig(model_name=self.gpt_model, temperature=temperature, max_tokens=max_tokens)
        request = Request(messages=messages, config=config)

        while patience > 0:
            patience -= 1
            try:
                response = self.server.evaluate(request)
                if response.success:
                    prediction = response.content.strip()
                    if prediction and prediction != "" and "Failed to obtain answer via API" not in prediction:
                        ret = self.can_infer(prediction, options)
                        data["gpt_prediction"] = ret
                        return data
                else:
                    eval_logger.error(f"Error from server: {response.error_message}")

            except Exception as e:
                # some model may output repetitive answer, which ChatGPT will throw an error.
                eval_logger.error(e)

                if sleep_time > 0:
                    time.sleep(sleep_time)
        return data

    def build_prompt(self, question, options, prediction):
        options_prompt = ""
        for key, item in options.items():
            options_prompt += f"{key}. {item}\n"
        tmpl = (
            "You are an AI assistant who will help me to match "
            "an answer with several options of a single-choice question. "
            "You are provided with a question, several options, and an answer, "
            "and you need to find which option is most similar to the answer. "
            "If the meaning of all options are significantly different from the answer, output Z. "
            "Your should output a single uppercase character in A, B, C, D (if they are valid options), and Z. \n"
            "Example 1: \n"
            "Question: What is the main object in image?\nOptions: A. teddy bear B. rabbit C. cat D. dog\n"
            "Answer: a cute teddy bear\nYour output: A\n"
            "Example 2: \n"
            "Question: What is the main object in image?\nOptions: A. teddy bear B. rabbit C. cat D. dog\n"
            "Answer: Spider\nYour output: Z\n"
            "Example 3: \n"
            "Question: {}\nOptions: {}\nAnswer: {}\nYour output: "
        )
        return tmpl.format(question, options_prompt, prediction)
