# OpenAI Compatible vs Async OpenAI Models

This document explains the differences between the `openai_compatible` and `async_openai` model implementations in `lmms-eval`. While both serve to interact with OpenAI-compatible API endpoints (e.g., GPT-4o, vLLM, SGLang), they differ significantly in their architecture and capabilities.

## Quick Comparison

| Feature | `openai_compatible` | `async_openai` |
| :--- | :--- | :--- |
| **Path** | `lmms_eval/models/chat/openai_compatible.py` | `lmms_eval/models/chat/async_openai.py` |
| **Concurrency** | Thread-based (`ThreadPoolExecutor`) | Async-based (`asyncio`) |
| **Client** | Synchronous `openai` client | Asynchronous `AsyncOpenAI` client |
| **Tool Use / MCP** | ❌ No | ✅ **Yes** (via MCP Client) |
| **Video Handling** | Basic | Advanced (Specific logic for Qwen2.5-VL/Qwen3-VL) |
| **Performance** | Good for moderate concurrency | Better for high concurrency (IO-bound) |

## Detailed Breakdown

### 1. `openai_compatible` (Synchronous)

The **`openai_compatible`** model is a standard implementation for evaluating Vision-Language Models (VLMs) hosted on OpenAI-compatible endpoints.

*   **Implementation**: It uses the synchronous `openai` Python client.
*   **Parallelism**: To achieve speedup, it uses python's `concurrent.futures.ThreadPoolExecutor`. This allows multiple requests to be sent in parallel threads.
*   **Use Case**: Best suited for standard evaluation tasks (VQA, Captioning) where the interaction is a single-turn "User Request -> Model Response" flow.
*   **Limitations**: It does not support complex agentic workflows involving tool calling loops or the Model Context Protocol (MCP).

### 2. `async_openai` (Asynchronous)

The **`async_openai`** model is a more advanced implementation designed for high-performance and complex interaction patterns.

*   **Implementation**: It uses the `AsyncOpenAI` client and Python's `asyncio` library.
*   **Parallelism**: It uses `asyncio` tasks and semaphores to manage concurrency, which is generally more efficient for IO-bound network operations than threading.
*   **Tool Use & MCP**: This is the **key differentiator**. `async_openai` has built-in support for the **Model Context Protocol (MCP)**.
    *   It can detect when a model wants to call a tool (`finish_reason == "tool_calls"`).
    *   It executes the tool via an MCP client.
    *   It feeds the tool output back to the model in a multi-turn loop until a final answer is generated.
*   **Advanced Features**: It contains specific logic for handling newer video models and their specific API requirements (e.g., Qwen2.5-VL video optimizations).

## When to Use Which?

*   **Use `async_openai` if**:
    *   You are evaluating **Agentic capabilities** or **Tool Use**.
    *   You need to use an **MCP Server**.
    *   You are evaluating advanced video models like **Qwen2.5-VL** that might benefit from its specific handling.
    *   You require very high concurrency that might bottleneck a thread pool.

*   **Use `openai_compatible` if**:
    *   You are running standard, static benchmarks (e.g., MMMU, MathVista) on a standard inference endpoint.
    *   You prefer a simpler execution flow for debugging basic model connections.
