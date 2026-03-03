# lmms-eval (fork)

Personal fork of [EvolvingLMMs-Lab/lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval).
Adds custom tasks, models, evaluation tooling, and prompt strategies on top of the upstream framework.
For installation, quickstart, and general usage refer to the original repo.

---

## What's Added

- [New Tasks](#new-tasks) — 6 new benchmarks
- [Extended Tasks](#extended-tasks) — 5 upstream tasks with additional metrics/variants
- [New Models](#new-models) — `qwen_agent`, `thyme`, `vllm_generate`
- [Prompt Configs](#prompt-configs) — reusable reasoning strategies
- [Demo Tools](#demo-tools) — result browser, diff viewer, chat demo
- [Evaluation Scripts](#evaluation-scripts) — ready-to-run scripts per task/model

---

## New Tasks

### ZeroBench (`zerobench`)

Multi-image QA benchmark. Each question spans multiple images; the model must reason across them to produce a final answer wrapped in `{curly braces}`.

**Dataset**: `jonathan-roberts1/zerobench`
**Metrics**: `exact_match` (bracket extraction) + `llm_judge` (GPT-4 semantic correctness)

```bash
python -m lmms_eval --model qwen2_5_vl \
  --model_args pretrained=Qwen/Qwen2.5-VL-7B-Instruct \
  --tasks zerobench --batch_size 1
```

A `zerobench_sub` variant runs on a smaller subset for faster iteration.

---

### CountBenchQA (`countbenchqa`)

Visual counting task. Answers are either a number (2–10) or a letter choice (A–I); the evaluator normalizes both forms before scoring.

**Dataset**: `vikhyatk/CountBenchQA`
**Metrics**: `exact_match` + `llm_judge`

```bash
python -m lmms_eval --model qwen2_5_vl \
  --model_args pretrained=Qwen/Qwen2.5-VL-7B-Instruct \
  --tasks countbenchqa --batch_size 1
```

---

### FrozenLake VQA (`frozenlake_vqa`)

Spatial navigation and grid-reasoning task built on FrozenLake environments rendered as images.

**Dataset**: `ohjoonhee/FrozenLakeVQA-32x32-1k`
**Metrics**: `accuracy`

```bash
python -m lmms_eval --model qwen2_5_vl \
  --model_args pretrained=Qwen/Qwen2.5-VL-7B-Instruct \
  --tasks frozenlake_vqa --batch_size 1
```

---

### Pix2Fact (`pix2fact`)

Image-to-fact generation task: given an image, the model enumerates factual claims about it.

**Metrics**: custom (`pix2fact_process_results`)

```bash
python -m lmms_eval --model qwen2_5_vl \
  --model_args pretrained=Qwen/Qwen2.5-VL-7B-Instruct \
  --tasks pix2fact --batch_size 1
```

---

### SimpleVQA (`simplevqa`)

Short-answer VQA with English and Chinese sub-tasks. Scores are aggregated with `weight_by_size`.

**Task group**: `simplevqa` → `simplevqa_en`, `simplevqa_cn`
**Metrics**: `is_correct`, `accuracy_given_attempted`, `f1`

```bash
python -m lmms_eval --model qwen2_5_vl \
  --model_args pretrained=Qwen/Qwen2.5-VL-7B-Instruct \
  --tasks simplevqa --batch_size 1
```

---

### WorldVQA (`worldvqa`)

World/geography knowledge VQA. Questions require cultural and factual knowledge about locations and landmarks worldwide.

**Dataset**: `moonshotai/WorldVQA`
**Metrics**: `worldvqa_accuracy`

```bash
python -m lmms_eval --model qwen2_5_vl \
  --model_args pretrained=Qwen/Qwen2.5-VL-7B-Instruct \
  --tasks worldvqa --batch_size 1
```

---

## Extended Tasks

These upstream tasks have been modified with additional metrics, variants, or model support.

### HR-Bench (`hrbench`)

Added thinking and lookback variants, agent/tool-calling support, and per-resolution sub-tasks.

| Task | Description |
|---|---|
| `hrbench` | Group: 4K + 8K standard |
| `hrbench4k` / `hrbench8k` | Standard 4096px / 8192px splits |
| `hrbench4k_think` / `hrbench8k_think` | With chain-of-thought system prompt |
| `hrbench4k_lookback` / `hrbench8k_lookback` | With lookback reasoning |

**Metrics**: `single` (single-image score), `cross` (cross-image score), `average`
**Judge**: GPT-3.5-turbo (configurable via `MODEL_VERSION` env var)

```bash
# Standard
python -m lmms_eval --model qwen2_5_vl \
  --model_args pretrained=Qwen/Qwen2.5-VL-7B-Instruct \
  --tasks hrbench4k --batch_size 1

# With think-first prompting
python -m lmms_eval --model vllm \
  --model_args model=Qwen/Qwen3-VL-8B-Thinking \
  --tasks hrbench \
  --lmms_eval_specific_kwargs configs/prompts/think_first_answer_concise.yaml
```

---

### V*Star Bench (`vstar_bench`)

Added LLM judge metric, category-level sub-tasks, and an agentic evaluation mode via `qwen_agent`.

| Task | Description |
|---|---|
| `vstar_bench` | Standard (parse + LLM judge) |
| `vstar_bench_direct_attributes` | Direct attribute queries only |
| `vstar_bench_relative_position` | Spatial relation queries only |

**New metrics**: `vstar_overall_acc`, `vstar_overall_llm_judge`

```bash
# Standard
python -m lmms_eval --model qwen2_5_vl \
  --model_args pretrained=Qwen/Qwen2.5-VL-7B-Instruct \
  --tasks vstar_bench --batch_size 1

# Agentic (serves vLLM locally, then runs qwen_agent)
bash scripts/vstar_bench/qwen3vl_agent.sh
```

---

### Blink (`blink`)

Added a GPT-4 LLM judge alongside the existing parse-based accuracy metric.

**New metric**: `blink_llm_judge`

---

### MMStar (`mmstar`)

Added object-counting (OC) variants with three prompt configurations, an LLM judge metric, and `doc_to_messages` support.

| Task | Description |
|---|---|
| `mmstar` | Standard |
| `mmstar_oc` | Object-counting prompt |
| `mmstar_oc_v3` | OC v3 text-format variant |
| `mmstar_ko` | Korean variant |
| `mmstar_qwen` | Qwen-specific prompt |

Prompt configs: `configs/prompts/mmstar_oc_v1.yaml`, `v2.yaml`, `v3.yaml`

---

### RealWorldQA (`realworldqa`)

Added LLM judge metric and improved prediction post-processing for short-answer extraction.

**New metric**: `realworldqa_llm_judge`

---

## New Models

### `qwen_agent`

Wraps a vLLM-served Qwen3-VL model inside the `qwen-agent` agentic loop. On each question, the model can invoke tools (default: `image_zoom_in_tool`) and observe results before producing a final answer.

**Requires**: a running vLLM server (`--enable-auto-tool-choice --tool-call-parser hermes`) and `qwen-agent` installed.

```bash
pip install qwen-agent

# Serve the model
vllm serve Qwen/Qwen3-VL-2B-Instruct \
  --port 10630 --enable-auto-tool-choice --tool-call-parser hermes

# Evaluate
python -m lmms_eval \
  --model qwen_agent \
  --model_args model_version=Qwen/Qwen3-VL-2B-Instruct,base_url=http://localhost:10630/v1,api_key=EMPTY \
  --tasks vstar_bench --batch_size 1
```

See `scripts/vstar_bench/qwen3vl_agent.sh` for a self-contained script that starts the server, waits for readiness, runs evaluation, then shuts the server down.

---

### `thyme`

**THinking YieldEd Modules.** Extends `Qwen2.5-VL` with an iterative chain-of-thought loop that can write and execute Python image-processing code inside a sandboxed environment.

On each step the model may emit `<code>...</code>` blocks (PIL/OpenCV operations: crop, resize, rotate, contrast). The sandbox executes them, returns the result, and the model continues reasoning. A `<answer>` tag terminates the loop.

**Key params**:
- `max_iterations` — reasoning iterations before fallback (default: 3)
- `max_retry` — generation retries on malformed output (default: 2)
- `verbose` — log per-step token counts and timing

```bash
python -m lmms_eval \
  --model thyme \
  --model_args pretrained=Qwen/Qwen2.5-VL-7B-Instruct,max_iterations=3 \
  --tasks hrbench4k --batch_size 1
```

See `scripts/hrbench/thyme.sh`.

---

### `vllm_generate`

Variant of `vllm_chat` that calls vLLM's `generate()` API instead of `chat()`. Useful for models that need manual `multi_modal_data` + `mm_processor_kwargs` construction (e.g., Qwen3-VL with complex video inputs).

```bash
python -m lmms_eval \
  --model vllm_generate \
  --model_args model=Qwen/Qwen3-VL-7B-Instruct \
  --tasks mme --batch_size 1
```

---

## Prompt Configs

Reusable YAML files in `configs/prompts/` that inject a system prompt and optional pre/post prompts into any task via `--lmms_eval_specific_kwargs`.

```bash
python -m lmms_eval ... \
  --lmms_eval_specific_kwargs configs/prompts/think_first.yaml
```

| File | Strategy |
|---|---|
| `think_first.yaml` | Plan → step-wise visual inspection → verify → synthesize |
| `think_first_v2.yaml` | Concise plan variant |
| `think_first_v3.yaml` | Aggressive verification variant |
| `think_first_answer_concise.yaml` | `think_first` + brief final answer requirement |
| `r1_think_system.yaml` | `<think>...</think>` R1-style reasoning block |
| `r1_think_system_example.yaml` | Same with a few-shot example |
| `atomic_perception_v2.yaml` | Interleaved `<think>/<query>/<resp>` perception loop |
| `mmstar_oc_v1/v2/v3.yaml` | Object-counting prompts for MMStar |
| `qwen3vl_hrbench.yaml` | HR-Bench specific Qwen3-VL system prompt |
| `qwen3vl_hrbench_pre.yaml` | Same with prepended instruction |
| `qwen3vl_vstar.yaml` | V*Star Bench specific Qwen3-VL system prompt |
| `qwen3vl_vstar_tool_in_user.yaml` | Tool instruction injected into user turn |
| `qwen3vl_frozen.yaml` | FrozenLake specific Qwen3-VL system prompt |

The YAML schema is:

```yaml
system_prompt: |
  <full system prompt text>
pre_prompt: ""   # prepended to each question
post_prompt: ""  # appended to each question
```

---

## Demo Tools

### Result Browser (`demo/lmms_eval_demo/`)

Flask app for browsing `--log_samples` JSONL output files. Shows thumbnails, pagination, and JSON field search.

```bash
python demo/lmms_eval_demo/app.py --data output/my_run/samples_mme.jsonl
```

Task-specific launchers: `mmstar_demo.sh`, `hrbench_demo.sh`, `vstarbench_demo.sh`, `zerobench_sub_demo.sh`, `worldvqa_demo.sh`, `charxiv_reasoning_demo.sh`.

---

### Results Diff Viewer (`demo/results_diff_viewer/`)

Flask app for side-by-side comparison of two evaluation result files. Aligns samples by `doc_id`, supports field filtering (`==`, `!=`, `>`, `<`, `contains`).

```bash
python demo/results_diff_viewer/app.py \
  --file1 output/run_a/samples.jsonl \
  --file2 output/run_b/samples.jsonl
```

---

### Qwen3-VL Chat Demo (`demo/Qwen3-VL-Chat-Demo/`)

Gradio chat interface for interactive Qwen3-VL evaluation with thinking toggle.

```bash
pip install -r demo/Qwen3-VL-Chat-Demo/requirements.txt
python demo/Qwen3-VL-Chat-Demo/app.py
```

---

### Attention & Heatmap Visualization (`demo/attn.py`, `demo/heatmap.py`)

Standalone scripts for visualizing model attention maps and token-level heatmaps over input images.

---

## Evaluation Scripts

`scripts/` contains ready-to-run bash scripts organized by task, each targeting a specific model/strategy combination. All scripts source a `.env` file for API keys and environment variables.

```
scripts/
├── zerobench/
│   ├── qwen3vl_base.sh             # standard inference
│   ├── qwen3vl_inst_base.sh        # instruction-tuned base
│   ├── qwen3vl_think_first.sh      # think_first prompt
│   ├── qwen3vl_think_first_v2.sh
│   └── qwen3vl_think_first_v3.sh
├── vstar_bench/
│   ├── qwen3vl_base.sh
│   ├── qwen3vl_agent.sh            # agentic (serves+evaluates+kills vLLM)
│   ├── qwen3vl_think_first.sh
│   ├── qwen3vl_base_sft.sh         # SFT checkpoint
│   └── ...
├── hrbench/
│   ├── qwen3vl_base.sh
│   ├── thyme.sh                    # Thyme model
│   ├── qwen3vl_agent.sh
│   └── ...
├── mmstar/ realworldqa/ simplevqa/
│   worldvqa/ viewspatial/ pix2fact/
│   mme_realworld/ blink/ chartqa/ ...
└── slurm/                          # SLURM sbatch wrappers
```

**Naming convention**:
- `_base` — direct inference, no special prompting
- `_inst_base` — instruction-tuned variant
- `_sft` — fine-tuned checkpoint
- `_think_first` — with `think_first` prompt config
- `_agent` — `qwen_agent` model (tool-calling loop)
- `_v2`, `_v3` — alternative prompt/config versions
- `seq.sh` — sequential (single-GPU, no accelerate)

---

## MCP Server Example

`examples/mcp_server/qwen_image_tools_server.py` implements a [Model Context Protocol](https://modelcontextprotocol.io) server that exposes image manipulation tools (zoom, crop) consumable by `qwen_agent` during evaluation.

```bash
python examples/mcp_server/qwen_image_tools_server.py
```

`examples/qwen3vl_image_tool.py` shows direct tool invocation with Qwen3-VL outside the eval loop.

---

## Syncing with Upstream

```bash
git fetch upstream
git merge upstream/main
```

Conflicts, when they occur, are typically limited to `README.md` (this file) and task `utils.py` files where upstream added new metrics.
