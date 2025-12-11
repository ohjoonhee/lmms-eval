from flask import Flask, render_template_string
import json
import os

app = Flask(__name__)

# Absolute path to the cache file
CACHE_FILE = "tmp/lmms_eval_cache/zerobench-qwen3-brain-eye-eyeparam/eval_cache/Qwen3BrainEyeVLLM_Qwen3BrainEyeVLLM_702c994cae574a63aa3c7c360ff74ca84bebe3c685ea7584a5e0a8536ed42d85/zerobench_rank0_world_size1.jsonl"

TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cache Viewer</title>
    <style>
        body { font-family: sans-serif; margin: 20px; background-color: #f4f4f9; }
        h1 { color: #333; }
        .container { max-width: 900px; margin: 0 auto; }
        .item { background: #fff; border: 1px solid #ddd; margin-bottom: 20px; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
        .doc-id { font-weight: bold; margin-bottom: 10px; color: #555; border-bottom: 1px solid #eee; padding-bottom: 5px; }
        .response { white-space: pre-wrap; background-color: #f8f8f8; padding: 15px; border-radius: 5px; font-family: monospace; color: #333; line-height: 1.5; overflow-x: auto; }
        .file-path { font-size: 0.9em; color: #666; margin-bottom: 20px; word-break: break-all; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Cache Viewer</h1>
        <div class="file-path"><strong>File:</strong> {{ file_path }}</div>
        {% for item in items %}
            <div class="item">
                <div class="doc-id">Doc ID: {{ item.doc_id }}</div>
                <div class="response">{{ item.response }}</div>
            </div>
        {% endfor %}
    </div>
</body>
</html>
"""


@app.route("/")
def index():
    items = []
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                for line in f:
                    if line.strip():
                        try:
                            items.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
        except Exception as e:
            return f"Error reading file: {e}"
    else:
        return f"File not found: {CACHE_FILE}"

    return render_template_string(TEMPLATE, items=items, file_path=CACHE_FILE)


if __name__ == "__main__":
    # Run on all interfaces to allow access if needed, though localhost is fine
    app.run(debug=True, host="0.0.0.0", port=5000)
