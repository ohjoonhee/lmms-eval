"""Results Diff Viewer - Compare two JSONL result files side-by-side."""

import argparse
import json
import os
from pathlib import Path

from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

# In-memory state
state: dict = {
    "data1": {},  # doc_id -> sample dict
    "data2": {},
    "aligned_ids": [],  # doc_ids present in both files
    "filtered_ids": [],  # currently filtered subset
    "fields1": [],
    "fields2": [],
}


def load_jsonl(path: str) -> dict[int, dict]:
    """Load a JSONL file and index by doc_id."""
    samples = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            doc_id = obj.get("doc_id")
            if doc_id is not None:
                samples[doc_id] = obj
    return samples


def collect_fields(samples: dict[int, dict]) -> list[str]:
    """Collect all unique field names across samples, with dot-notation for nested dicts."""
    fields: set[str] = set()
    for sample in samples.values():
        _collect_keys(sample, "", fields)
    return sorted(fields)


def _collect_keys(obj: dict, prefix: str, out: set[str]) -> None:
    for key, val in obj.items():
        full_key = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"
        out.add(full_key)
        if isinstance(val, dict):
            _collect_keys(val, full_key, out)


def get_nested_value(obj: dict, field: str):
    """Get a value from a dict, supporting dot notation for nested access."""
    parts = field.split(".")
    current = obj
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    return current


def apply_filter(samples: dict[int, dict], field: str, op: str, value: str) -> set[int]:
    """Return doc_ids that match the filter condition."""
    matching = set()
    for doc_id, sample in samples.items():
        raw = get_nested_value(sample, field)
        if raw is None:
            continue
        raw_str = str(raw)
        try:
            raw_num = float(raw_str)
            val_num = float(value)
            is_numeric = True
        except (ValueError, TypeError):
            is_numeric = False

        match = False
        if op == "==":
            match = raw_str == value or (is_numeric and raw_num == val_num)
        elif op == "!=":
            match = raw_str != value and (not is_numeric or raw_num != val_num)
        elif op == ">" and is_numeric:
            match = raw_num > val_num
        elif op == "<" and is_numeric:
            match = raw_num < val_num
        elif op == "contains":
            match = value in raw_str

        if match:
            matching.add(doc_id)
    return matching


@app.route("/")
def index():
    return render_template(
        "index.html",
        default_path1=app.config.get("DEFAULT_PATH1", ""),
        default_path2=app.config.get("DEFAULT_PATH2", ""),
        default_score_key=app.config.get("DEFAULT_SCORE_KEY", ""),
    )


@app.route("/api/load", methods=["POST"])
def load_files():
    body = request.get_json()
    path1 = body.get("path1", "")
    path2 = body.get("path2", "")

    if not os.path.isfile(path1):
        return jsonify({"error": f"File not found: {path1}"}), 400
    if not os.path.isfile(path2):
        return jsonify({"error": f"File not found: {path2}"}), 400

    state["data1"] = load_jsonl(path1)
    state["data2"] = load_jsonl(path2)
    common_ids = sorted(set(state["data1"].keys()) & set(state["data2"].keys()))
    state["aligned_ids"] = common_ids
    state["filtered_ids"] = list(common_ids)
    state["fields1"] = collect_fields(state["data1"])
    state["fields2"] = collect_fields(state["data2"])

    return jsonify({
        "count": len(common_ids),
        "fields1": state["fields1"],
        "fields2": state["fields2"],
        "total1": len(state["data1"]),
        "total2": len(state["data2"]),
    })


@app.route("/api/sample/<int:index>")
def get_sample(index: int):
    ids = state["filtered_ids"]
    if not ids or index < 0 or index >= len(ids):
        return jsonify({"error": "Index out of range"}), 400
    doc_id = ids[index]
    return jsonify({
        "index": index,
        "total": len(ids),
        "doc_id": doc_id,
        "left": state["data1"].get(doc_id, {}),
        "right": state["data2"].get(doc_id, {}),
    })


@app.route("/api/filter", methods=["POST"])
def filter_samples():
    body = request.get_json()
    filters = body.get("filters", [])

    # Start with all aligned doc_ids
    result_ids = set(state["aligned_ids"])

    for f in filters:
        side = f.get("side")  # "left" or "right"
        field = f.get("field", "")
        op = f.get("op", "==")
        value = f.get("value", "")

        if not field:
            continue

        data = state["data1"] if side == "left" else state["data2"]
        matching = apply_filter(data, field, op, value)
        result_ids &= matching

    state["filtered_ids"] = sorted(result_ids)
    return jsonify({"count": len(state["filtered_ids"])})


@app.route("/api/reset_filter", methods=["POST"])
def reset_filter():
    state["filtered_ids"] = list(state["aligned_ids"])
    return jsonify({"count": len(state["filtered_ids"])})


def _preload(path1: str, path2: str) -> None:
    """Pre-load two JSONL files into state at startup."""
    state["data1"] = load_jsonl(path1)
    state["data2"] = load_jsonl(path2)
    common_ids = sorted(set(state["data1"].keys()) & set(state["data2"].keys()))
    state["aligned_ids"] = common_ids
    state["filtered_ids"] = list(common_ids)
    state["fields1"] = collect_fields(state["data1"])
    state["fields2"] = collect_fields(state["data2"])
    print(f"Pre-loaded {len(state['data1'])} (left) and {len(state['data2'])} (right) samples, {len(common_ids)} aligned.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Results Diff Viewer")
    parser.add_argument("--path1", help="Path to left JSONL file (pre-load on startup)")
    parser.add_argument("--path2", help="Path to right JSONL file (pre-load on startup)")
    parser.add_argument("--score_key", help="Dot-notation key to display as score badge (e.g. vstar_overall_llm_judge.score)")
    parser.add_argument("--port", type=int, default=5001)
    args = parser.parse_args()

    if args.path1:
        app.config["DEFAULT_PATH1"] = args.path1
    if args.path2:
        app.config["DEFAULT_PATH2"] = args.path2
    if args.score_key:
        app.config["DEFAULT_SCORE_KEY"] = args.score_key
    if args.path1 and args.path2:
        _preload(args.path1, args.path2)

    app.run(debug=True, port=args.port)
