from typing import List, Optional, Union, Tuple
import os
import base64
import tempfile
import uuid
import shutil
from io import BytesIO

from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.protocol import ChatMessages

try:
    from qwen_agent.agents import Assistant
    from qwen_agent.llm.schema import ASSISTANT, FUNCTION
    from qwen_agent.utils.output_beautify import TOOL_CALL_S, TOOL_CALL_E, TOOL_RESULT_S, TOOL_RESULT_E, THOUGHT_S, ANSWER_S
except ImportError:
    eval_logger.warning("Failed to import qwen_agent; Please install it via `pip install qwen-agent`")


def typewriter_print(messages: List[dict], text: str) -> str:
    full_text = ""
    content = []
    for msg in messages:
        if msg["role"] == ASSISTANT:
            if msg.get("reasoning_content"):
                assert isinstance(msg["reasoning_content"], str), "Now only supports text messages"
                content.append(f'{THOUGHT_S}\n{msg["reasoning_content"]}')
            if msg.get("content"):
                assert isinstance(msg["content"], str), "Now only supports text messages"
                content.append(f'{ANSWER_S}\n{msg["content"]}')
            if msg.get("function_call"):
                content.append(f'{TOOL_CALL_S} {msg["function_call"]["name"]}\n{msg["function_call"]["arguments"]}')
        elif msg["role"] == FUNCTION:
            content.append(f'{TOOL_RESULT_S} {msg["name"]}\n{msg["content"]}')
        else:
            raise TypeError
    if content:
        full_text = "\n".join(content)
        # print(full_text[len(text) :], end="", flush=True)

    return full_text


@register_model("qwen_agent")
class QwenAgent(lmms):
    is_simple = False

    def __init__(
        self,
        model_version: str = "Qwen/Qwen3-VL-2B-Instruct",
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        tools: Optional[List[str]] = None,
        system_message: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()

        self.model_version = model_version
        self.base_url = base_url
        self.api_key = api_key

        # Default tools if none provided
        if tools is None:
            self.tools = ["image_zoom_in_tool"]
        else:
            self.tools = tools

        if system_message is None:
            self.system_message = None
        else:
            with open(system_message, "r") as f:
                self.system_message = f.read().strip()

        # Initialize the agent
        llm_cfg = {
            "model_type": "qwenvl_oai",
            "model": self.model_version,
            "model_server": self.base_url,
            "api_key": self.api_key,
            "generate_cfg": {"top_p": 0.8, "top_k": 20, "temperature": 0.7, "repetition_penalty": 1.0, "presence_penalty": 1.5},
        }

        self.agent = Assistant(
            llm=llm_cfg,
            function_list=self.tools,
            system_message=self.system_message,
        )

        # Temporary directory for saving images
        self.temp_dir = os.path.join(tempfile.gettempdir(), f"qwen_agent_images_{uuid.uuid4()}")
        os.makedirs(self.temp_dir, exist_ok=True)

    def __del__(self):
        if hasattr(self, "temp_dir") and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _resolve_model_name_for_cache(self) -> str:
        """
        Best-effort resolution of a human-readable model identifier for cache naming.
        Checks common attributes; falls back to class name.
        """
        return "QwenAgent/" + self.model_version

    def _save_image(self, image: Image.Image) -> str:
        """Save PIL image to temp file and return path."""
        img_id = str(uuid.uuid4())
        img_path = os.path.join(self.temp_dir, f"{img_id}.png")
        image.save(img_path)
        return img_path

    def make_one_request(self, request: Instance) -> Tuple[list[dict], dict]:
        """
        Build OpenAI-style messages and per-request sampling params from an Instance.
        Returns (messages, params_dict). Does not mutate input.
        """
        ctx, doc_to_messages, gen_kwargs, doc_id, task, split = request.arguments
        raw_messages = doc_to_messages(self.task_dict[task][split][doc_id])
        chat_messages = ChatMessages(messages=raw_messages)
        # Copy to avoid side-effects across threads
        _gen = dict(gen_kwargs or {})
        _gen.setdefault("max_new_tokens", 4096)
        _gen.setdefault("temperature", 0)
        _gen.setdefault("top_p", 0.95)

        params = {
            "temperature": _gen["temperature"],
            "max_tokens": _gen["max_new_tokens"],
            "top_p": _gen["top_p"],
        }

        messages = chat_messages.to_qwen3_agent_openai_messages(temp_dir=self.temp_dir)

        return messages, params

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []
        self.load_cache()
        res, requests = self.get_response_from_cache(requests)
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Qwen Agent Responding")

        # TODO: Add Batch support (if applicable)

        for request in requests:
            messages, params = self.make_one_request(request)

            with open("MESSAGE.txt", "w") as f:
                f.write(str(messages))

            response_plain_text = ""
            for ret_messages in self.agent.run(messages):
                # `ret_messages` will contain all subsequent messages, consisting of interleaved assistant messages and tool responses
                # response_plain_text = multimodal_typewriter_print(ret_messages, response_plain_text)
                response_plain_text = typewriter_print(ret_messages, response_plain_text)

            final_response = response_plain_text

            self.add_request_response_to_cache(request, final_response)
            pbar.update(1)
            res.append(final_response)

        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood not implemented for Qwen Agent")

    def generate_until_multi_round(self, requests: List[Instance]) -> List[str]:
        raise NotImplementedError("Multi-round generation not implemented for Qwen Agent")
