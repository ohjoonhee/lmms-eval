import base64
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from typing import List, Optional, Tuple, Union, Dict

import numpy as np
from accelerate import Accelerator, DistributedType
from decord import VideoReader, cpu
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from openai import OpenAI

try:
    from vllm import LLM, SamplingParams
except ImportError:
    vllm = None

WORKERS = int(os.getenv("WORKERS", "32"))


@register_model("qwen3_brain_eye_vllm")
class Qwen3BrainEyeVLLM(lmms):
    """
    Qwen3 Brain-Eye Pipeline using vLLM.
    Uses two vLLM instances: one for the "Brain" (Qwen3) and one for the "Eye" (Qwen3-VL).
    """

    def __init__(
        self,
        brain_model: str = "Qwen/Qwen3-1.7B",
        eye_model: str = "Qwen/Qwen3-VL-2B-Instruct",
        brain_args: Optional[Union[str, Dict]] = None,
        eye_args: Optional[Union[str, Dict]] = None,
        batch_size: int = 1,
        max_frame_num: int = 32,
        trust_remote_code: Optional[bool] = True,
        min_image_pixels: int = 28,
        **kwargs,
    ) -> None:
        super().__init__()
        self.brain_model_name = brain_model
        self.eye_model_name = eye_model
        self.max_frame_num = max_frame_num
        self.min_image_pixels = min_image_pixels
        self.batch_size_per_gpu = int(batch_size)

        # Parse model-specific arguments
        self.brain_args = self._parse_args(brain_args)
        self.eye_args = self._parse_args(eye_args)

        # Set up vLLM environment
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

        accelerator = Accelerator()
        self.accelerator = accelerator
        self._rank = self.accelerator.local_process_index
        self._world_size = self.accelerator.num_processes

        if self.accelerator.num_processes > 1:
            # For multi-GPU/node, we might need external launcher, but running 2 vLLMs in one process
            # is tricky with distributed setups. Assuming single node with multiple GPUs for now.
            pass

        # Initialize Brain Client (OpenAI)
        eval_logger.info(f"Initializing Brain Client for: {brain_model}")
        api_key = self.brain_args.get("api_key", "EMPTY")
        base_url = self.brain_args.get("base_url", "http://localhost:8000/v1")

        self.brain_client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

        # Initialize Eye LLM
        eval_logger.info(f"Initializing Eye Model: {eye_model}")
        self.eye_llm = LLM(
            model=eye_model,
            trust_remote_code=trust_remote_code if trust_remote_code is not None else True,
            max_model_len=32768,
            gpu_memory_utilization=0.4,
            seed=1,
            **self.eye_args,
        )

    def _parse_args(self, args: Union[str, Dict, None]) -> Dict:
        if args is None:
            return {}
        if isinstance(args, dict):
            return args
        if isinstance(args, str):
            try:
                # Try parsing as JSON
                return json.loads(args)
            except json.JSONDecodeError:
                # Try parsing as key=value,key2=value2
                parsed = {}
                for pair in args.split(","):
                    if "=" in pair:
                        k, v = pair.split("=", 1)
                        # Try to convert to int/float/bool
                        if v.lower() == "true":
                            v = True
                        elif v.lower() == "false":
                            v = False
                        else:
                            try:
                                v = int(v)
                            except ValueError:
                                try:
                                    v = float(v)
                                except ValueError:
                                    pass
                        parsed[k.strip()] = v
                return parsed
        return {}

    def _maybe_resize_image(self, img: Image.Image) -> Image.Image:
        if self.min_image_pixels <= 0:
            return img
        if min(img.size) <= 0:
            raise ValueError(f"Invalid image dimensions: {img.size}")
        if min(img.size) >= self.min_image_pixels:
            return img
        scale = self.min_image_pixels / min(img.size)
        new_size = tuple(int(dim * scale) for dim in img.size)
        return img.resize(new_size, Image.BICUBIC)

    def encode_image(self, image: Union[Image.Image, str]):
        if isinstance(image, str):
            img = Image.open(image).convert("RGB")
        else:
            img = image.copy()
        img = self._maybe_resize_image(img)
        output_buffer = BytesIO()
        img.save(output_buffer, format="PNG")
        byte_data = output_buffer.getvalue()
        return base64.b64encode(byte_data).decode("utf-8")

    def encode_video(self, video_path):
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frame_num = len(vr)
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, self.max_frame_num, dtype=int)
        if total_frame_num - 1 not in uniform_sampled_frames:
            uniform_sampled_frames = np.append(uniform_sampled_frames, total_frame_num - 1)
        frame_idx = uniform_sampled_frames.tolist()
        frames = vr.get_batch(frame_idx).asnumpy()
        base64_frames = []
        for frame in frames:
            img = Image.fromarray(frame)
            img = self._maybe_resize_image(img)
            output_buffer = BytesIO()
            img.save(output_buffer, format="PNG")
            base64_frames.append(base64.b64encode(output_buffer.getvalue()).decode("utf-8"))
        return base64_frames

    def flatten(self, input):
        new_list = []
        for i in input:
            if isinstance(i, (list, tuple)):
                new_list.extend(i)
            else:
                new_list.append(i)
        return new_list

    def generate_until(self, requests) -> List[str]:
        res = []
        self.load_cache()
        res, requests = self.get_response_from_cache(requests)
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Brain-Eye Responding")

        # Define the Eye tool
        eye_tool = {
            "type": "function",
            "function": {
                "name": "eye_look",
                "description": "Use the Eye tool to look at the image and answer a question about it.",
                "parameters": {"type": "object", "properties": {"question": {"type": "string", "description": "The question to ask the Eye about the image."}}, "required": ["question"]},
            },
        }
        tools = [eye_tool]

        for req in requests:
            contexts, gen_kwargs, doc_to_visual, doc_id, task, split = req.arguments

            # Prepare visuals
            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            if None in visuals:
                visuals = []
                imgs = []
            else:
                visuals = self.flatten(visuals)
                imgs = []
                with ThreadPoolExecutor(max_workers=WORKERS) as executor:
                    futures = []
                    for visual in visuals:
                        if isinstance(visual, str) and any(visual.endswith(ext) for ext in [".mp4", ".avi", ".mov", ".flv", ".wmv"]):
                            futures.append(executor.submit(self.encode_video, visual))
                        elif isinstance(visual, (str, Image.Image)):
                            futures.append(executor.submit(self.encode_image, visual))
                    for f in futures:
                        imgs.append(f.result())

            # --- Step 1: Brain Planning ---
            system_prompt = (
                "You are a helpful assistant with access to a vision tool called 'eye_look'. "
                "The user will ask a question that may require visual information. "
                "The image is not directly provided to you. "
                "Instead, you can use the 'eye_look' tool to ask questions about the image if needed. "
                "You MUST decompose the question into smaller subquestions and ask them one by one, so that the 'eye_look' tool can answer them with a few words directly. "
                "The 'eye_look' tool may make some mistakes, so you should try to verify the answer again with different questions. "
            )

            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": contexts}]

            # Brain generation params
            temperature = gen_kwargs.get("temperature", 0.6)
            top_p = gen_kwargs.get("top_p", 0.95)
            max_tokens = gen_kwargs.get("max_new_tokens", 1024)

            # 1st Brain Pass
            response = self.brain_client.chat.completions.create(
                model=self.brain_model_name,
                messages=messages,
                tools=tools,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )

            response_message = response.choices[0].message
            messages.append(response_message.model_dump())
            reasoning_message = response_message.reasoning_content
            # final_answer = response_message.content or ""
            final_answer = "<think>" + reasoning_message

            eval_logger.info(f"Brain 1st response: {final_answer}")

            # --- Step 2: Tool Execution ---
            for _ in range(10):
                if response_message.tool_calls:
                    for tool_call in response_message.tool_calls:
                        if tool_call.function.name == "eye_look":
                            try:
                                args = json.loads(tool_call.function.arguments)
                                eye_query = args.get("question", "Describe the image.")
                                eval_logger.info(f"Brain requested tool: {eye_query}")
                                
                                final_answer += "\n\n<tool_query>" + eye_query + "</tool_query>"

                                # Prepare Eye messages
                                eye_messages = [{"role": "user", "content": []}]
                                for img in self.flatten(imgs):
                                    eye_messages[0]["content"].append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}})
                                eye_messages[0]["content"].append({"type": "text", "text": eye_query})

                                # Eye generation params
                                eye_params = SamplingParams(max_tokens=512, temperature=0.7, top_p=0.8, top_k=20, presence_penalty=1.5, repetition_penalty=1.0) 

                                # Eye Pass
                                eye_output = self.eye_llm.chat(messages=eye_messages, sampling_params=eye_params)
                                eye_response_text = eye_output[0].outputs[0].text
                                eval_logger.info(f"Eye response: {eye_response_text}")

                                messages.append(
                                    {
                                        "role": "tool",
                                        "content": eye_response_text,
                                        "tool_call_id": tool_call.id,
                                    }
                                )
                                final_answer += "\n\n<tool_response>" + eye_response_text + "</tool_response>"
                            except Exception as e:
                                eval_logger.error(f"Error executing tool: {e}")
                                messages.append(
                                    {
                                        "role": "tool",
                                        "content": f"Error executing tool: {str(e)}",
                                        "tool_call_id": tool_call.id,
                                    }
                                )

                    # 2nd Brain Pass
                    response = self.brain_client.chat.completions.create(
                        model=self.brain_model_name,
                        messages=messages,
                        tools=tools,
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=max_tokens,
                    )

                    response_message = response.choices[0].message
                    # if response_message.content:
                    if response.choices[0].finish_reason == "stop":
                        break
                    messages.append(response_message.model_dump())
                    reasoning_message = response_message.reasoning_content
                    final_answer += "\n\n" + reasoning_message

                # final_response = self.brain_client.chat.completions.create(
                #     model=self.brain_model_name,
                #     messages=messages,
                #     tools=tools,
                #     temperature=temperature,
                #     top_p=top_p,
                #     max_tokens=max_tokens,
                # )
            final_answer += "\n</think>\n\n" + (response_message.content or "")

            self.add_request_response_to_cache(req, final_answer)

            res.append(final_answer)
            pbar.update(1)

        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood not supported for Brain-Eye pipeline")

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("Multi-round generation not supported yet")
