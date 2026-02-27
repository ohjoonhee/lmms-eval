# New Model Guide

To evaluate a model with `lmms_eval`, you implement a wrapper class that subclasses `lmms_eval.api.model.lmms`. This guide walks through the full process.

## Architecture Overview

```
    ╭──────────────╮                              ╭─────────────╮
    │  Model Dev   │                              │  Task Dev   │
    ╰──────┬───────╯                              ╰──────┬──────╯
           │                                             │
           ▼                                             ▼
 ┌─────────────────────┐                    ┌─────────────────────┐
 │                     │                    │                     │
 │  Implement lmms     │                    │  Create task YAML   │
 │  wrapper            │                    │  + utils.py         │
 │                     │                    │                     │
 │  Core methods:      │                    │  Preferred:         │
 │  · generate_until   │                    │  · doc_to_messages  │
 │  · loglikelihood    │                    │                     │
 │                     │                    │  Legacy:            │
 │  Register           │                    │  · doc_to_visual    │
 │  ModelManifest in   │                    │  · doc_to_text      │
 │  ModelRegistryV2    │                    │                     │
 └─────────┬───────────┘                    └──────────┬──────────┘
           │                                           │
           │           ╭ ─ ─ ─ ─ ─ ─ ─ ╮              │
           ╰──────────▶   Evaluator     ◀──────────────╯
                       │   contract     │
                        ╰ ─ ─ ─ ┬ ─ ─ ─╯
                                │
                                ▼
                  ┌───────────────────────┐
                  │                       │
                  │  Unified Instance     │
                  │  requests             │
                  │                       │
                  │  Model inference      │
                  │                       │
                  │  process_results      │
                  │                       │
                  │  metrics aggregation  │
                  │                       │
                  └───────────────────────┘
```

**Model Dev** implements the left side; **Task Dev** implements the right side. The evaluator runtime wires them together - your model never needs to know which task is calling it, and vice versa.

## Model Types

| Type | Location | Input method | Recommendation |
|------|----------|-------------|----------------|
| **Chat** | `models/chat/` | `doc_to_messages` - structured messages with roles and content types | **Use this** |
| **Simple** (legacy) | `models/simple/` | `doc_to_visual` + `doc_to_text` - plain text with `<image>` placeholders | Legacy only |

## Setup

```sh
git clone https://github.com/<YOUR-USERNAME>/lmms-eval.git
cd lmms-eval
git checkout -b <model-type>
pip install -e .

# Create your model file
touch lmms_eval/models/chat/<my_model>.py     # recommended
touch lmms_eval/models/simple/<my_model>.py   # legacy
```

Reference implementations: `lmms_eval/models/chat/qwen2_5_vl.py` (chat) and `lmms_eval/models/simple/instructblip.py` (simple).

## Core Methods

All models must subclass `lmms_eval.api.model.lmms` and implement two methods. Each receives a list of `Instance` objects (defined in [`lmms_eval.api.instance`](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/lmms_eval/api/instance.py)) whose `.args` carry the request payload.

### `generate_until`

Open-ended generation. The model produces text given an input prompt + media.

**`Instance.args` for chat models** (5 elements):

| Element | Type | Description |
|---------|------|-------------|
| `doc_to_messages` | `Callable` | Function that converts a doc into structured `ChatMessages` |
| `gen_kwargs` | `dict` | Generation config: `max_new_tokens`, `temperature`, `until`, etc. |
| `doc_id` | `int` | Index into the dataset split |
| `task` | `str` | Task name (used to look up the dataset via `self.task_dict`) |
| `split` | `str` | Dataset split name |

**`Instance.args` for simple models** (6 elements):

| Element | Type | Description |
|---------|------|-------------|
| `contexts` | `str` | Formatted question text (may contain `<image>` tokens) |
| `gen_kwargs` | `dict` | Generation config |
| `doc_to_visual` | `Callable` | Function that returns a list of media (PIL images, video paths, etc.) |
| `doc_id` | `int` | Index into the dataset split |
| `task` | `str` | Task name |
| `split` | `str` | Dataset split name |

Returns `list[str]` - one generated string per request.

### `loglikelihood`

Scoring for multiple-choice tasks. The model computes the log-probability of a target continuation given a context.

**`Instance.args`** (6 elements):

| Element | Type | Description |
|---------|------|-------------|
| `contexts` | `str` | Formatted question text |
| `doc_to_target` | `Callable` | Function that extracts the answer continuation from the doc |
| `doc_to_visual` | `Callable` | Function that returns media |
| `doc_id` | `int` | Index into the dataset split |
| `task` | `str` | Task name |
| `split` | `str` | Dataset split name |

Returns `list[tuple[float, bool]]` - `(log_prob, is_greedy)` per request, where `is_greedy` is `True` if the target would be produced by greedy decoding.

## Registration

Register your model so `lmms_eval` can find it via `--model <name>`.

```python
from lmms_eval.api.registry import register_model

@register_model("my_model")
class MyModel(lmms):
    is_simple = False  # chat model (recommended)
    # is_simple = True  # simple model (legacy, default)
```

Then add the entry in `lmms_eval/models/__init__.py`:

```python
# Recommended (ModelRegistryV2 manifest)
from lmms_eval.models.registry_v2 import ModelManifest

MODEL_REGISTRY_V2.register_manifest(
    ModelManifest(
        model_id="my_model",
        chat_class_path="lmms_eval.models.chat.my_model.MyModel",
    )
)

# Legacy (still supported)
AVAILABLE_CHAT_TEMPLATE_MODELS["my_model"] = "MyModel"
```

For external plugin packages, prefer Python entry-points (`lmms_eval.models`) over `LMMS_EVAL_PLUGINS`.

## Complete Example (Chat Model)

```python
from lmms_eval.api.registry import register_model
from lmms_eval.api.model import lmms
from lmms_eval.api.instance import Instance
from lmms_eval.protocol import ChatMessages
import torch


@register_model("my_image_model")
class MyImageModel(lmms):
    is_simple = False

    def __init__(self, pretrained: str, device: str = "cuda", **kwargs):
        super().__init__()
        self.device = device
        self.model = load_your_model(pretrained)
        self.processor = load_your_processor(pretrained)

    def generate_until(self, requests: list[Instance]) -> list[str]:
        results = []
        for request in requests:
            doc_to_messages, gen_kwargs, doc_id, task, split = request.args

            # Build structured messages from the doc
            doc = self.task_dict[task][split][doc_id]
            raw_messages = doc_to_messages(doc)
            messages = ChatMessages(messages=raw_messages)

            # Extract media and format prompt
            images, videos, audios = messages.extract_media()
            hf_messages = messages.to_hf_messages()
            text = self.processor.apply_chat_template(hf_messages)

            # Run inference
            inputs = self.processor(
                text=text, images=images, return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=gen_kwargs.get("max_new_tokens", 128),
                    temperature=gen_kwargs.get("temperature", 0.0),
                    do_sample=gen_kwargs.get("do_sample", False),
                )

            response = self.processor.decode(
                outputs[0], skip_special_tokens=True
            )
            results.append(response)
        return results

    def loglikelihood(
        self, requests: list[Instance]
    ) -> list[tuple[float, bool]]:
        results = []
        for request in requests:
            contexts, doc_to_target, doc_to_visual, doc_id, task, split = (
                request.args
            )
            # Compute log-probability of the target continuation
            # given the context + visual inputs.
            # ...
        return results
```

For video and audio models the pattern is identical - the only difference is which media you extract from `messages.extract_media()`. See `lmms_eval/models/chat/qwen2_5_vl.py` for a production-quality reference.

## Key Notes

```python
@register_model("my_video_model")
class MyVideoModel(lmms):
    is_simple = False
    
    def __init__(self, pretrained: str, max_frames: int = 8, **kwargs):
        super().__init__()
        self.max_frames = max_frames
        # Initialize video model
        
    def generate_until(self, requests: list[Instance]) -> list[str]:
        results = []
        for request in requests:
            doc_to_messages, gen_kwargs, doc_id, task, split = request.args
            doc = self.task_dict[task][split][doc_id]
            messages = doc_to_messages(doc)
            
            # Extract video frames
            images, videos, audios = messages.extract_media()
            text_prompt = ""
            for message in messages:
                if message.type == "text":
                    text_prompt += message["text"]
            
            # Process video frames and generate response
            # ...
        return results
    
    def extract_frames(self, video_path, max_frames):
        # Extract frames from video file
        # Return list of PIL Images or tensors
        pass
```

### Audio Model Extension

For audio models, adapt the pattern to handle audio inputs:

```python
@register_model("my_audio_model")
class MyAudioModel(lmms):
    is_simple = False
    
    def __init__(self, pretrained: str, sample_rate: int = 16000, **kwargs):
        super().__init__()
        self.sample_rate = sample_rate
        # Initialize audio model
        
    def generate_until(self, requests: list[Instance]) -> list[str]:
        results = []
        for request in requests:
            doc_to_messages, gen_kwargs, doc_id, task, split = request.args
            doc = self.task_dict[task][split][doc_id]
            messages = doc_to_messages(doc)
            
            images, videos, audios = messages.extract_media()
            text_prompt = ""
            for message in messages:
                if message.type == "text":
                    text_prompt += message["text"]
            
            # Process audio and generate response
            # ...
        return results
    
    def load_audio(self, audio_path, sample_rate):
        # Load audio file and resample if needed
        # Return audio tensor or array
        pass
```

## Key Implementation Notes

1. **Image Models**: Handle visual inputs through PIL Images or tensors, typically support single or multiple images
2. **Video Models**: Extract frames from videos, handle temporal relationships
3. **Audio Models**: Process audio waveforms, handle different sample rates and formats

Remember to:
- Handle different input modalities in the `doc_to_messages` function
- Process model-specific tokens (e.g., `<image>`, `<video>`, `<audio>`)
- Implement both `generate_until` and `loglikelihood` methods if your model supports both generation and multiple-choice tasks
- Follow the existing model implementations in `lmms_eval/models/chat/` for reference

## OpenAI and API Models

If you are working with OpenAI-compatible API models (like standard OpenAI endpoints, vLLM, SGLang, etc.), please refer to [OpenAI Models Comparison](openai_models_comparison.md) for a detailed comparison of the `openai_compatible` and `async_openai` implementations.
- Implement both `generate_until` and `loglikelihood` if your model supports generation and multiple-choice tasks
- Handle different modalities (image, video, audio) via the `ChatMessages` protocol
- Follow existing implementations in `lmms_eval/models/chat/` for patterns around batching, device management, and error handling
