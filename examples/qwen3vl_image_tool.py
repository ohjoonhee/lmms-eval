import base64
import json
import math
import os
import uuid
from io import BytesIO
from math import ceil, floor
from typing import List, Union

import dotenv
import qwen_agent
import requests
from openai import OpenAI
from PIL import Image

# from qwen_agent.utils.utils import extract_images_from_messages

dotenv.load_dotenv()

client = OpenAI(
    base_url="https://api.fireworks.ai/inference/v1",
    api_key=os.getenv("FIREWORKS_API_KEY"),
)


# Image resizing functions (copied from qwen-vl-utils)
def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def smart_resize(height: int, width: int, factor: int = 32, min_pixels: int = 56 * 56, max_pixels: int = 12845056) -> tuple[int, int]:
    """Smart resize image dimensions based on factor and pixel constraints"""
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def maybe_resize_bbox(left, top, right, bottom, img_width, img_height):
    """Resize bbox to ensure it's valid"""
    left = max(0, left)
    top = max(0, top)
    right = min(img_width, right)
    bottom = min(img_height, bottom)

    height = bottom - top
    width = right - left
    if height < 32 or width < 32:
        center_x = (left + right) / 2.0
        center_y = (top + bottom) / 2.0
        ratio = 32 / min(height, width)
        new_half_height = ceil(height * ratio * 0.5)
        new_half_width = ceil(width * ratio * 0.5)
        new_left = floor(center_x - new_half_width)
        new_right = ceil(center_x + new_half_width)
        new_top = floor(center_y - new_half_height)
        new_bottom = ceil(center_y + new_half_height)

        # Ensure the resized bbox is within image bounds
        new_left = max(0, new_left)
        new_top = max(0, new_top)
        new_right = min(img_width, new_right)
        new_bottom = min(img_height, new_bottom)

        new_height = new_bottom - new_top
        new_width = new_right - new_left

        if new_height > 32 and new_width > 32:
            return [new_left, new_top, new_right, new_bottom]
    return [left, top, right, bottom]


def image_zoom_in(bbox_2d, label, img_idx, messages):

    def extract_images_from_messages(messages):
        images = []
        for message in messages:
            role = getattr(message, "role", None) or message.get("role")
            content = getattr(message, "content", None) or message.get("content")
            if role == "user" or role == "tool":
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "image_url":
                            images.append(item["image_url"]["url"])
        return images

    images = extract_images_from_messages(messages)
    print("len(images)", len(images))

    # Ensure work directory exists - using a local temp dir for example
    work_dir = "workspace/image_zoom_output"
    os.makedirs(work_dir, exist_ok=True)

    try:
        # open image, currently only support the first image
        image_arg = images[img_idx]
        if image_arg.startswith("data:image/jpeg;base64,"):
            # Base64 encoded string
            image_string = image_arg[len("data:image/jpeg;base64,") :]
            image = Image.open(BytesIO(base64.b64decode(image_string)))
        else:
            if image_arg.startswith("file://"):
                image_arg = image_arg[len("file://") :]

            if image_arg.startswith("http"):
                response = requests.get(image_arg)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content))
            elif os.path.exists(image_arg):
                image = Image.open(image_arg)
            else:
                # Fallback if just basename is provided, check work_dir or cwd
                possible_path = os.path.join(work_dir, image_arg)
                if os.path.exists(possible_path):
                    image = Image.open(possible_path)
                else:
                    image = Image.open(image_arg)  # Try direct path again or fail

        # Validate and potentially resize bbox
        img_width, img_height = image.size
        # bbox is [x1, y1, x2, y2] normalized to 1000x1000
        rel_x1, rel_y1, rel_x2, rel_y2 = bbox_2d
        abs_x1, abs_y1, abs_x2, abs_y2 = rel_x1 / 1000.0 * img_width, rel_y1 / 1000.0 * img_height, rel_x2 / 1000.0 * img_width, rel_y2 / 1000.0 * img_height

        validated_bbox = maybe_resize_bbox(abs_x1, abs_y1, abs_x2, abs_y2, img_width, img_height)

        left, top, right, bottom = validated_bbox

        # Crop the image
        cropped_image = image.crop((left, top, right, bottom))

        # Resize according to smart_resize logic
        new_w, new_h = smart_resize((right - left), (bottom - top), factor=32, min_pixels=256 * 32 * 32)
        cropped_image = cropped_image.resize((new_w, new_h), resample=Image.BICUBIC)

        output_filename = f"{uuid.uuid4()}.png"
        output_path = os.path.abspath(os.path.join(work_dir, output_filename))
        cropped_image.save(output_path)

        return {"image_path": output_path}
    except Exception as e:
        return {"error": f"Tool Execution Error {str(e)}"}


# Your actual tool implementation
def get_weather(location, unit="celsius"):
    # In production, call your weather API here
    return {"temperature": 72, "condition": "sunny", "unit": unit}


tools = [
    # {
    #     "type": "function",
    #     "function": {
    #         "name": "get_weather",
    #         "description": "Get the current weather in a given location",
    #         "parameters": {
    #             "type": "object",
    #             "properties": {
    #                 "location": {
    #                     "type": "string",
    #                     "description": "The city and state, e.g. San Francisco, CA",
    #                 },
    #                 "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
    #             },
    #             "required": ["location"],
    #         },
    #     },
    # },
    {
        "type": "function",
        "function": {
            "name": "image_zoom_in",
            "description": "Zoom in on a specific region of an image by cropping it based on a bounding box (bbox) and an optional object label",
            "parameters": {
                "type": "object",
                "properties": {
                    "bbox_2d": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 4,
                        "maxItems": 4,
                        "description": "The bounding box of the region to zoom in, as [x1, y1, x2, y2], where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner",
                    },
                    "label": {"type": "string", "description": "The name or label of the object in the specified bounding box"},
                    "img_idx": {"type": "number", "description": "The index of the zoomed-in image (starting from 0)"},
                },
                "required": ["bbox_2d", "label", "img_idx"],
            },
        },
    },
]


def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


image_path = "data/24.jpg"
messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant with access to functions. Use them if required.",
    },
    {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encode_image(image_path)}",
                },
            },
            {
                "type": "text",
                "text": "What's written in the back plate of a car in the image?",
            },
        ],
    },
]

log_messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant with access to functions. Use them if required.",
    },
    {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": os.path.abspath(image_path),
                },
            },
            {
                "type": "text",
                "text": "What's written in the back plate of a car in the image?",
            },
        ],
    },
]

# Use a model capable of tool calling.
# "accounts/fireworks/models/llama-v3p3-70b-instruct" is used here as an example.
model_name = "accounts/fireworks/models/qwen3-vl-30b-a3b-instruct"

os.makedirs("debug", exist_ok=True)
MAX_TURNS = 10

for i in range(MAX_TURNS):
    print(f"Turn {i + 1}/{MAX_TURNS}: Sending request to model: {model_name}...")
    completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )

    message = completion.choices[0].message
    # with open(f"debug/message_{i}.txt", "w") as f:
    #     f.write(str(message))

    # Check for native tool calls
    if message.tool_calls:
        print("Tool calls found:")
        tool_call = message.tool_calls[0]
        print(f"Function: {tool_call.function.name}")
        print(f"Arguments: {tool_call.function.arguments}")

        if tool_call.function.name == "get_weather":
            # Parse arguments and call your function
            function_args = json.loads(tool_call.function.arguments)
            function_response = get_weather(**function_args)
            log_function_response = function_response
        elif tool_call.function.name == "image_zoom_in":
            function_args = json.loads(tool_call.function.arguments)
            # image_zoom_in requires 'messages' for context
            function_response = image_zoom_in(**function_args, messages=messages)
            _image_path = function_response["image_path"]  # Function returns the image_path of the result saved
            _encoded_image = encode_image(_image_path)

            # Prepare responses for log (path) and api (base64)
            log_function_response = {"type": "image_url", "image_url": {"url": _image_path}}
            function_response = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_encoded_image}"}}

        else:
            function_response = {"error": "Unknown function"}
            log_function_response = function_response

        # print(f"Function response: {function_response}")
        # with open(f"debug/function_response_{i}.txt", "w") as f:
        #     f.write(str(function_response))

        # Send tool response back to model
        assistant_message = message.model_dump(exclude_none=True)
        messages.append(assistant_message)  # Add assistant's tool call
        log_messages.append(assistant_message)

        messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": [function_response]})
        log_messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": [log_function_response]})

    else:
        print("No native tool calls returned.")
        print("Final Response:")
        print(message.content)

        with open("debug/final_response.jsonl", "w") as f:
            json.dump(log_messages, f)
        # with open("debug/final_response.txt", "w") as f:
        #     f.write(str(log_messages))
        break
