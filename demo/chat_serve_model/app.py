# SPDX-License-Identifier: Apache-2.0

import argparse
import base64
import mimetypes
import os

import gradio as gr
from openai import OpenAI

# Argument parser setup
parser = argparse.ArgumentParser(description="Chatbot Interface with Customizable Parameters")
parser.add_argument("--model-url", type=str, default="http://localhost:8000/v1", help="Model URL")
parser.add_argument("-m", "--model", type=str, default="Qwen/Qwen3-VL-32B-Thinking-FP8", help="Model name for the chatbot")
parser.add_argument("--temp", type=float, default=0.8, help="Temperature for text generation")
parser.add_argument("--stop-token-ids", type=str, default="", help="Comma-separated stop token IDs")
parser.add_argument("--host", type=str, default=None)
parser.add_argument("--port", type=int, default=8001)

# Parse the arguments
args = parser.parse_args()

# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = args.model_url

# Create an OpenAI client to interact with the API server
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def predict(message, history, thinking_mode):
    # Convert chat history to OpenAI format
    history_openai_format = [{"role": "system", "content": "You are a great ai assistant."}]
    for human, assistant in history:
        # Extract text if human message is a dict (multimodal)
        human_text = human["text"] if isinstance(human, dict) else human
        history_openai_format.append({"role": "user", "content": human_text})
        history_openai_format.append({"role": "assistant", "content": assistant})

    # Handle multimodal message
    if isinstance(message, dict) and "files" in message and len(message["files"]) > 0:
        content = [{"type": "text", "text": message["text"]}]
        for file_path in message["files"]:
            # Check if it's an image
            mime_type, _ = mimetypes.guess_type(file_path)
            if mime_type and mime_type.startswith("image"):
                base64_image = encode_image(file_path)
                content.append({"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}})
        history_openai_format.append({"role": "user", "content": content})
    else:
        # Text only
        text_message = message["text"] if isinstance(message, dict) else message
        history_openai_format.append({"role": "user", "content": text_message})

    # Create a chat completion request and send it to the API server
    stream = client.chat.completions.create(
        model=args.model,  # Model name to use
        messages=history_openai_format,  # Chat history
        temperature=args.temp,  # Temperature for text generation
        stream=True,  # Stream response
        extra_body={
            "repetition_penalty": 1,
            "stop_token_ids": [int(id.strip()) for id in args.stop_token_ids.split(",") if id.strip()] if args.stop_token_ids else [],
            # "chat_template_kwargs": {"enable_thinking": thinking_mode},
        },
    )

    # Read and return generated text from response stream
    partial_message = ""
    for chunk in stream:
        try:
            partial_message += chunk.choices[0].delta.reasoning_content or ""
        except AttributeError:
            pass
        partial_message += chunk.choices[0].delta.content or ""
        yield partial_message


# Create and launch a chat interface with Gradio
gr.ChatInterface(
    predict,
    multimodal=True,
    additional_inputs=[gr.Checkbox(label="Thinking Mode", value=False)],
).queue().launch(server_name=args.host, server_port=args.port, share=True)
