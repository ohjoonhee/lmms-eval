# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

lmms-eval is a unified evaluation framework for Large Multimodal Models (LMMs) supporting 190+ tasks across image, video, and audio modalities. It is a fork of lm-evaluation-harness adapted for multimodal evaluation. Supports 70+ models including local (HuggingFace, vLLM, SGLang) and API-based (OpenAI, Claude, Gemini).

## Common Commands

```bash
# Environment setup (uses uv, NOT pip)
uv sync                                    # Create environment from uv.lock
uv add <package>                           # Add dependency
uv run <command>                           # Run any command

# Running evaluations
python -m lmms_eval --model qwen2_5_vl \
  --model_args pretrained=Qwen/Qwen2.5-VL-3B-Instruct \
  --tasks mme,mmmu --batch_size 8 --device cuda:0

# Multi-GPU evaluation with accelerate
accelerate launch --num_processes 8 --main_process_port 12345 -m lmms_eval \
  --model llava --model_args pretrained=path --tasks mme --batch_size 1

# List available tasks
python -m lmms_eval --tasks list
python -m lmms_eval --tasks list_with_num  # With sample counts (downloads data)

# Code quality
uv run ruff format .                       # Format code
uv run ruff check .                        # Lint check
uv run ruff check . --fix                  # Auto-fix lint issues
uv run pyright                             # Type checking
uv run pytest                              # Run tests
uv run pytest tests/test_file.py -k name   # Run specific test

# Regression testing (after PRs)
python3 tools/regression.py
```

## Key CLI Arguments

- `--model`: Model type (e.g., `qwen2_5_vl`, `llava_hf`, `gpt4v`, `claude`, `vllm`, `async_openai`)
- `--model_args`: Comma-separated model params (e.g., `pretrained=path,batch_size=1`)
- `--tasks`: Comma-separated task names or groups
- `--batch_size`: Batch size for evaluation (start small, increase until OOM; `"auto"` is NOT implemented for most models)
- `--limit`: Limit docs per task (int or float 0.0-1.0 for percentage)
- `--device`: Device (`cuda`, `cuda:0`, `cpu`, `mps`)
- `--output_path`: Results directory (string like `dir/file.jsonl` or `dir/`)
- `--log_samples`: Save per-document outputs (must be used with `--output_path`)
- `--cache_requests`: Cache requests (`true`, `refresh`, `delete`). Files stored under `lmms_eval/cache/.cache` or `$LM_HARNESS_CACHE_PATH`
- `--include_path`: Add custom task YAML directory
- `--gen_kwargs`: Override generation params for all `generate_until` tasks (format: `key1=val1,key2=val2`)
- `--num_fewshot`: Number of few-shot examples (not well tested)
- `--predict_only`: Generate outputs without computing metrics (use with `--log_samples`)
- `--system_instruction`: System instruction string to prepend to prompts
- `--apply_chat_template`: Apply chat template to prompts (optionally specify template name)
- `--seed`: Set seeds for random/numpy/torch (comma-separated or single int, default `0,1234,1234`)
- `--wandb_args`: W&B logging (e.g., `project=test-project,name=test-run`)
- `--write_out`: Debug flag to print prompt and gold target for first document of each task
- `--show_config`: Print full TaskConfig for each task after evaluation

## Environment Variables

```bash
export OPENAI_API_KEY="..."         # For GPT-4V and OpenAI-compatible models
export ANTHROPIC_API_KEY="..."      # For Claude models
export GOOGLE_API_KEY="..."         # For Gemini models
export HF_HOME="..."               # HuggingFace cache directory
export HF_TOKEN="..."              # HuggingFace authentication
export HF_HUB_ENABLE_HF_TRANSFER="1"  # Faster HF downloads
export LMMS_EVAL_USE_CACHE=True    # Enable JSONL response caching
export LMMS_EVAL_HOME="..."        # Cache root (default: ~/.cache/lmms-eval)
```

## Architecture

### Core Evaluation Flow

```
simple_evaluate() → TaskManager loads tasks → Model processes requests → Metrics computed
     ↓                    ↓                         ↓                        ↓
  evaluator.py      tasks/__init__.py         models/*/           api/metrics.py
```

### Registry Pattern

Pre-commit hooks run automatically on `git commit` (requires `pre-commit install`):

- **Black**: Python formatting (line-length=240)
- **isort**: Import sorting (profile=black)
- Config: `.pre-commit-config.yaml`
- Run manually: `pre-commit run --all-files`
- Always run before pushing to avoid CI failures

## Code Style

- PEP 8 naming: snake_case functions/variables, PascalCase classes, UPPER_SNAKE_CASE constants
- Type hints required for all code
- Public APIs must have docstrings
- Use f-strings for formatting
- Follow existing patterns exactly

## Architecture Overview (v0.6)

lmms-eval is a unified evaluation framework for Large Multimodal Models supporting image, video, and audio tasks.

### Directory Structure

```
lmms_eval/
├── __main__.py          # CLI entry point
├── api/                 # Base classes: model.lmms, Instance, registry
├── models/
│   ├── __init__.py      # Registry (AVAILABLE_SIMPLE_MODELS, AVAILABLE_CHAT_TEMPLATE_MODELS, MODEL_ALIASES)
│   ├── registry_v2.py   # ModelManifest, ModelRegistryV2 (aliasing + typed resolution)
│   ├── chat/            # Chat models (recommended) - structured ChatMessages input
│   └── simple/          # Simple/legacy models - doc_to_visual + doc_to_text input
├── tasks/               # Task definitions (YAML config + utils.py per task, auto-registered)
├── protocol.py          # ChatMessages protocol (structured multimodal input)
├── entrypoints/         # Eval server (EvalClient, AsyncEvalClient, ServerArgs)
├── llm_judge/           # LLM-as-judge scoring providers
├── loggers/             # Result logging
└── tui/                 # Web UI (React + FastAPI)
```

### Evaluation Pipeline

```
Input (dataset + question) --> Model (inference) --> Judge (metrics)
                                     |
                                 Cache layer (hash of input + config + git_commit)
```

Key v0.6 features:
- **Eval as a Service**: HTTP server for async evaluation during training (`/evaluate`, `/jobs/{id}`, `/queue`)
- **Statistical Rigor**: Confidence intervals, clustered standard errors (`cluster_key`), paired model comparison
- **Model Registry V2**: Single model_id resolves to chat (preferred) or simple. Old names via aliases in `MODEL_ALIASES`.
- **Async Pipeline**: Decoupled inference + judging with cache for crash recovery and deduplication

### Model Types

| Type | Location | Input | Class flag |
|------|----------|-------|------------|
| **Chat** (recommended) | `models/chat/` | `doc_to_messages` -> `ChatMessages` | `is_simple = False` |
| **Simple** (legacy) | `models/simple/` | `doc_to_visual` + `doc_to_text` | `is_simple = True` |

Resolution: chat > simple (unless `force_simple=True`).

### Launch Command

```bash
python -m lmms_eval --model qwen2_5_vl --model_args pretrained=Qwen/Qwen2.5-VL-3B-Instruct,max_pixels=12845056,attn_implementation=sdpa --tasks mmmu,mme --batch_size 1 --limit 8 --device cuda:0
```

## How to Add a New Model

### 1. Create model file

```bash
touch lmms_eval/models/chat/my_model.py    # recommended
# or: touch lmms_eval/models/simple/my_model.py  (legacy only)
```

Reference implementations: `chat/qwen2_5_vl.py`, `chat/qwen3_vl.py`, `simple/instructblip.py`.

### 2. Implement the class

Subclass `lmms_eval.api.model.lmms`. Implement `generate_until` (required) and `loglikelihood` (for multiple-choice tasks).

```python
from lmms_eval.api.registry import register_model
from lmms_eval.api.model import lmms
from lmms_eval.protocol import ChatMessages

@register_model("my_model")
class MyModel(lmms):
    is_simple = False  # chat model

    def __init__(self, pretrained: str, device: str = "cuda", **kwargs):
        super().__init__()
        self.model = load_model(pretrained)
        self.processor = load_processor(pretrained)

    def generate_until(self, requests):
        results = []
        for req in requests:
            doc_to_messages, gen_kwargs, doc_id, task, split = req.args  # 5 elements for chat
            doc = self.task_dict[task][split][doc_id]
            messages = ChatMessages(messages=doc_to_messages(doc))
            images, videos, audios = messages.extract_media()
            text = self.processor.apply_chat_template(messages.to_hf_messages())
            # ... run inference ...
            results.append(response)
        return results

    def loglikelihood(self, requests):
        # For multiple-choice. Return list[(log_prob, is_greedy)]
        ...
```

### 3. Register in `lmms_eval/models/__init__.py`

```python
AVAILABLE_CHAT_TEMPLATE_MODELS = {
    ...
    "my_model": "MyModel",   # auto-constructs lmms_eval.models.chat.my_model.MyModel
}

# Optional: backward-compatible aliases
MODEL_ALIASES = {
    "my_model": ("old_name",),
}
```

### 4. Test

```bash
python -m lmms_eval --model my_model --model_args pretrained=org/model --tasks mme --limit 5 --batch_size 1
```

## How to Add a New Task

### 1. Create task directory

```
lmms_eval/tasks/my_task/
├── my_task.yaml     # Task config (auto-registered by filename)
└── utils.py         # doc_to_messages, process_results, aggregation functions
```

No manual registration needed - the framework scans `tasks/` for YAML files.

### 2. Write YAML config

```yaml
task: "my_task"
dataset_path: org/my-dataset
dataset_kwargs:
  token: True
test_split: test
output_type: generate_until    # or: loglikelihood, generate_until_multi_round

doc_to_messages: !function utils.my_doc_to_messages  # for chat models (recommended)
doc_to_visual: !function utils.my_doc_to_visual      # for simple models (legacy)
doc_to_text: !function utils.my_doc_to_text          # for simple models (legacy)
doc_to_target: "answer"

generation_kwargs:
  max_new_tokens: 1024
  temperature: 0

process_results: !function utils.my_process_results
metric_list:
  - metric: accuracy
    aggregation: !function utils.my_aggregate
    higher_is_better: true

# Model-specific prompt overrides (passed to doc_to_messages/doc_to_text)
lmms_eval_specific_kwargs:
  default:
    pre_prompt: ""
    post_prompt: "\nAnswer with the option's letter from the given choices directly."
  qwen3_vl:
    format: "qwen3_vl"
    pre_prompt: "Question: "
    post_prompt: "Answer with the option letter only."

# Model-specific generation overrides
model_specific_generation_kwargs:
  llava:
    image_aspect_ratio: original

metadata:
  - version: 0.0
```

#### Task Groups and Inheritance

Use `group` + `task` list to create task families (e.g., mmmu with 30+ subtasks):

```yaml
# lmms_eval/tasks/mmmu/mmmu.yaml (group definition)
group: mmmu
task:
- mmmu_val
- mmmu_test
```

Use `include` to share config across task variants (avoids duplication):

```yaml
# lmms_eval/tasks/textcaps/textcaps_val.yaml
task: "textcaps_val"
group: "textcaps_caption"
test_split: val
# ... task-specific overrides ...
include: _default_template_textcaps_yaml   # inherits shared config
```

#### lmms_eval_specific_kwargs

This mechanism lets tasks define model-specific prompt formatting. The `default` key is used for most models; add model-specific keys to override:

```yaml
lmms_eval_specific_kwargs:
  default:
    pre_prompt: ""
    post_prompt: "\nAnswer directly."
  qwen_vl:
    format: qwen_vl          # completely different format
  idefics2:
    pre_prompt: ""
    post_prompt: "\nAnswer:"  # different suffix
```

The kwargs are passed to `doc_to_messages(doc, lmms_eval_specific_kwargs=...)` and `doc_to_text(doc, lmms_eval_specific_kwargs=...)`. The framework selects the matching key based on the model being evaluated, falling back to `default`.

### 3. Implement utils.py

```python
def my_doc_to_messages(doc, lmms_eval_specific_kwargs=None):
    # lmms_eval_specific_kwargs contains model-specific prompt config
    # (e.g., pre_prompt, post_prompt) — selected by framework based on model
    kwargs = lmms_eval_specific_kwargs or {}
    pre = kwargs.get("pre_prompt", "")
    post = kwargs.get("post_prompt", "")

    messages = [{"role": "user", "content": []}]
    if doc.get("image"):
        messages[0]["content"].append({"type": "image", "url": doc["image"]})
    messages[0]["content"].append({"type": "text", "text": pre + doc["question"] + post})
    return messages

def my_process_results(doc, results):
    pred = results[0].strip()
    return {"accuracy": 1.0 if pred == doc["answer"] else 0.0}

def my_aggregate(results):
    return sum(results) / len(results)
```

### 4. Test

```bash
python -m lmms_eval --model qwen2_5_vl --tasks my_task --limit 8 --batch_size 1
```

## Commit Conventions

- Bug/feature from user report: `git commit --trailer "Reported-by:<name>"`
- Related to GitHub issue: `git commit --trailer "Github-Issue:#<number>"`
- NEVER mention `co-authored-by` or the tool used to create commits or PRs

## Pull Requests

- Focus on the high-level problem description and how it is solved
- Don't go into code specifics unless it adds clarity
- NEVER mention `co-authored-by` or the tool used to create the PR

- [lmms_eval/__main__.py](lmms_eval/__main__.py) - CLI entry point, argument parsing
- [lmms_eval/evaluator.py](lmms_eval/evaluator.py) - Core `simple_evaluate()` and `evaluate()` functions
- [lmms_eval/api/task.py](lmms_eval/api/task.py) - Task base class and configuration
- [lmms_eval/api/model.py](lmms_eval/api/model.py) - `lmms` base class for all models
- [lmms_eval/api/registry.py](lmms_eval/api/registry.py) - Registration decorators
- [lmms_eval/api/instance.py](lmms_eval/api/instance.py) - Request/response dataclasses
- [lmms_eval/api/metrics.py](lmms_eval/api/metrics.py) - Built-in metric and aggregation functions
- [lmms_eval/protocol.py](lmms_eval/protocol.py) - Message protocol for chat models

### Model Types

1. **Chat models** (`lmms_eval/models/chat/`) - Recommended for new models. Use `doc_to_messages` for interleaved text/image/video/audio. Set `is_simple = False`.
2. **Simple models** (`lmms_eval/models/simple/`) - Legacy type using `doc_to_visual` + `doc_to_text`. Set `is_simple = True` (default).

Reference implementations:
- Chat: `lmms_eval/models/chat/qwen2_5_vl.py`
- Simple: `lmms_eval/models/simple/instructblip.py`

### OpenAI-Compatible Models

Two implementations exist for OpenAI-compatible APIs:

| Feature | `openai_compatible` | `async_openai` |
|---------|-------------------|----------------|
| Concurrency | Thread-based | Async (asyncio) |
| Tool Use / MCP | No | Yes |
| Video Handling | Basic | Advanced (Qwen2.5-VL/Qwen3-VL) |
| Best for | Standard benchmarks | Agentic evals, high concurrency |

### Request Types

Models implement two core methods:

- **`generate_until(requests)`** - Free-form text generation. Chat models receive `(doc_to_messages, gen_kwargs, doc_id, task, split)`. Simple models receive `(contexts, gen_kwargs, doc_to_visual, doc_id, task, split)`.
- **`loglikelihood(requests)`** - Log probability of target conditioned on input. Returns `list[tuple[float, bool]]` where bool indicates if target is the greedy choice.

### Task Configuration

Tasks are defined via YAML in `lmms_eval/tasks/<task_name>/`:

```yaml
task: "task_name"
dataset_path: "huggingface/dataset"
dataset_kwargs:
  token: True
test_split: test
output_type: "generate_until"  # generate_until, loglikelihood, multiple_choice, generate_until_multi_round
doc_to_text: !function utils.doc_to_text        # Simple model text input
doc_to_visual: !function utils.doc_to_visual    # Simple model visual input
doc_to_messages: !function utils.doc_to_messages # Chat model interleaved input
doc_to_target: "answer"                          # Column name or !function
doc_to_choice: !function utils.doc_to_choice     # For multiple_choice tasks
generation_kwargs:
  max_new_tokens: 128
  temperature: 0
  do_sample: false
process_results: !function utils.process_results # Parallel (multi-GPU) post-processing
metric_list:
  - metric: exact_match
    aggregation: mean
    higher_is_better: true
lmms_eval_specific_kwargs:                       # Model-specific prompt configs
  default:
    pre_prompt: ""
    post_prompt: "\nAnswer the question using a single word or phrase."
metadata:
  - version: 0.0
```

Key YAML features:
- `!function utils.func_name` embeds Python functions from a `utils.py` in the same directory
- `include: base_template.yaml` inherits from another YAML
- `process_results` runs in parallel (multi-GPU); `aggregate_results` runs on rank 0

### Built-in Metrics

Metrics: `acc`, `acc_norm`, `acc_all`, `anls`, `acc_mutual_info`, `by_pass`, `exact_match`, `perplexity`, `word_perplexity`, `byte_perplexity`, `bits_per_byte`, `brier_score`, `matthews_corrcoef`, `f1`, `bleu`, `chrf`, `ter`

Aggregation functions: `mean`, `median`, `perplexity`, `weighted_perplexity`, `bits_per_byte`

### Response Caching

JSONL-based caching system for avoiding redundant API calls:

```bash
export LMMS_EVAL_USE_CACHE=True
```

- Cache key: `(task_name, doc_id)` - ensure stable doc_ids across runs
- Location: `$LMMS_EVAL_HOME/eval_cache/<model_hash>/`
- Files: `{task_name}_rank{rank}_world_size{world_size}.jsonl`
- Distributed-safe with per-rank files

In model's `generate_until`:
```python
self.load_cache()
cached, pending = self.get_response_from_cache(requests)
# process pending, call self.add_request_response_to_cache(req, out)
```

### Throughput Metrics

Chat models automatically log inference metrics at INFO level:
- **E2E**: Total latency (seconds)
- **TTFT**: Time to first token
- **TPOT**: Time per output token
- **Speed**: Tokens/second (1/TPOT)
- **Output Tokens**: Count of generated tokens

Supported: `sglang_runtime`, `vllm_chat`, `llava_hf_chat`, `openai_compatible_chat`, `qwen2_5_vl_chat`, `huggingface_chat`

### Programmatic API

```python
import lmms_eval

results = lmms_eval.simple_evaluate(
    model=lmm_obj,           # Your lmms subclass instance
    tasks=["mme", "mmmu"],
    num_fewshot=0,
    task_manager=lmms_eval.tasks.TaskManager(),  # Optional, for custom paths
)
```

### Task Categories (190+)

| Category | Count | Examples |
|----------|-------|---------|
| Image VQA & Understanding | 60+ | `mme`, `mmmu_val`, `ai2d`, `vqav2`, `realworldqa` |
| MMBench Family | 15+ | `mmbench_en`, `mmstar`, `mmvet`, `mmvetv2` |
| Multi-image | 15+ | `mmmu`, `muirbench`, `llava_interleave_bench` |
| Video Understanding | 25+ | `videomme`, `mvbench`, `egoschema`, `mlvu` |
| Long Video & Temporal | 10+ | `longvideobench`, `temporalbench`, `moviechat` |
| Audio & Speech | 20+ | `air_bench`, `librispeech`, `covost2`, `voicebench` |
| Document Understanding | 12+ | `docvqa`, `ocrbench`, `textvqa`, `infovqa` |
| Mathematical Reasoning | 12+ | `mathvista`, `mathvision`, `mathverse`, `aime` |
| Spatial & Grounding | 10+ | `refcoco`, `embspatial`, `vstar_bench` |
| Text-only Language | 15+ | `mmlu`, `gpqa`, `hellaswag`, `gsm8k` |

Use `python -m lmms_eval --tasks list` for the authoritative list.

## Development Rules

### Package Management
- ONLY use uv, NEVER pip
- FORBIDDEN: `uv pip install`, `@latest` syntax

### Code Quality
- Type hints required for all code
- Public APIs must have docstrings
- Line length: 88 chars maximum
- Run formatters before type checks

### Testing
- Framework: `uv run pytest`
- Async testing: use anyio, not asyncio
- New features require tests
- Bug fixes require regression tests

### Commits
```bash
# For bug fixes from user reports
git commit --trailer "Reported-by:<name>"

# For GitHub issues
git commit --trailer "Github-Issue:#<number>"
```

NEVER mention co-authored-by or tools used to create commits/PRs.

## Code Style

- PEP 8 naming: snake_case functions/variables, PascalCase classes, UPPER_SNAKE_CASE constants
- Use f-strings for formatting
- Early returns to avoid nesting
- Keep functions focused and small

## Adding New Components

### Adding a Model

1. Create file in `lmms_eval/models/chat/` (recommended) or `lmms_eval/models/simple/`
2. Inherit from `lmms` base class
3. Set `is_simple = False` for chat models (recommended), `True` for simple
4. Implement `generate_until()` and optionally `loglikelihood()`
5. Register with `@register_model("name")`
6. Import in `lmms_eval/models/__init__.py`
7. Reference: `lmms_eval/models/chat/qwen2_5_vl.py` (chat), `lmms_eval/models/simple/instructblip.py` (simple)

Chat model `generate_until` pattern:
```python
for request in requests:
    doc_to_messages, gen_kwargs, doc_id, task, split = request.args
    doc = self.task_dict[task][split][doc_id]
    messages = doc_to_messages(doc)
    images, videos, audios = messages.extract_media()
    # Process and generate...
```

### Adding a Task

1. Create directory `lmms_eval/tasks/<task_name>/`
2. Add `task.yaml` with configuration (see Task Configuration above)
3. Add `utils.py` for custom `doc_to_text`, `doc_to_visual`, `doc_to_messages`, `process_results`, aggregation functions
4. Use `!function utils.func_name` in YAML to reference Python functions
5. Use `include: base.yaml` to inherit from templates
6. For audio tasks: use `doc_to_audio` function to load audio from HF datasets
7. Reference tasks: MME (`lmms_eval/tasks/mme/`), SeedBench PPL (`lmms_eval/tasks/seedbench/`), MMSearch multi-round (`lmms_eval/tasks/mmsearch/`)

Notes:
- `process_results` executes in parallel (multi-GPU) - use for collecting/parsing outputs and GPT-4 judging
- `aggregate_results` executes on rank 0 - use for computing final scores
- `lmms_eval_specific_kwargs` defines model-specific prompts (default follows LLaVA format)
- Batch size > 1 may produce different scores for some models; use batch_size=1 for final benchmarking

## CI/CD Fix Order

1. Formatting (`ruff format`)
2. Type errors (`pyright`)
3. Linting (`ruff check`)
