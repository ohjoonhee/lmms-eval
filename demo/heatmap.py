#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Token Logprob Heatmap — JSONL + D3 Graph Tooltip
- Single JSONL OR folder with multiple JSONLs (Prev/Next, ←/→ navigation)
- Larger tooltip sized to graph; clamped to viewport
- Fixed x-scale per JSONL for consistent comparisons across tooltips
- Logprob value labels shown INSIDE each bar (right-aligned)
- NEW: Jump-to-page input (1-based) in multi-page mode
"""

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

from flask import Flask, abort, redirect, render_template_string, url_for
from markupsafe import escape

app = Flask(__name__)

INPUT_IS_DIR: bool = False
FILE_LIST: List[str] = []
PAGE_CACHE: Dict[str, Tuple[List[dict], dict]] = {}


# -------- Utilities --------
def load_jsonl(path: str) -> Tuple[List[str], List[Optional[int]], List[Dict[str, Any]]]:
    tokens, token_ids, logprobs = [], [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            tok = obj.get("token", "")
            if not isinstance(tok, str):
                tok = str(tok)
            tokens.append(tok)
            token_ids.append(obj.get("token_id"))
            meta = obj.get("logprob", {})
            if not isinstance(meta, dict):
                meta = {}
            logprobs.append(meta)
    return tokens, token_ids, logprobs


def min_max(vals: List[float]) -> Tuple[float, float]:
    if not vals:
        return 0.0, 1.0
    vmin, vmax = min(vals), max(vals)
    if abs(vmax - vmin) < 1e-9:
        vmax = vmin + 1e-9
    return vmin, vmax


def to_hsl_color(v: float, vmin: float, vmax: float) -> str:
    if vmax - vmin < 1e-9:
        vmax = vmin + 1e-9
    norm = (v - vmin) / (vmax - vmin)
    norm = max(0.0, min(1.0, norm))
    hue = 120.0 * norm
    return f"hsl({hue:.1f},85%,70%)"


def build_items_for_file(jsonl_path: str) -> Tuple[List[dict], dict]:
    tokens, token_ids, metas = load_jsonl(jsonl_path)
    if not tokens:
        raise ValueError("No tokens loaded.")

    items, vals = [], []
    all_logprobs = []  # for global x-scale
    newline_markers = {"\n", "\r\n", "\r", "Ċ"}

    for token_text, tid, meta in zip(tokens, token_ids, metas):
        candidates = []
        for k, v in meta.items():
            if isinstance(v, dict) and "logprob" in v:
                logp = float(v.get("logprob", 0.0))
                all_logprobs.append(logp)
                candidates.append(
                    {
                        "decoded_token": v.get("decoded_token", str(k)),
                        "logprob": logp,
                        "rank": int(v.get("rank", 999)),
                    }
                )

        if not candidates:
            lp = 0.0
        else:
            candidates.sort(key=lambda x: x["rank"])
            match = next((c for c in candidates if c["decoded_token"] == token_text), None)
            lp = match["logprob"] if match else candidates[0]["logprob"]
        vals.append(lp)

        # Newline handling
        if token_text in newline_markers:
            items.append({"type": "br"})
            continue

        if "\n" in token_text:
            parts = token_text.split("\n")
            for i, part in enumerate(parts):
                if part:
                    items.append(
                        {
                            "type": "token",
                            "text": escape(part),
                            "raw": part,
                            "lp": lp,
                            "alts": candidates,
                        }
                    )
                if i < len(parts) - 1:
                    items.append({"type": "br"})
            continue

        items.append(
            {
                "type": "token",
                "text": escape(token_text),
                "raw": token_text,
                "lp": lp,
                "alts": candidates,
            }
        )

    vmin, vmax = min_max(vals)
    gmin, gmax = min_max(all_logprobs)

    for it in items:
        if it.get("type") == "token":
            it["color"] = to_hsl_color(it["lp"], vmin, vmax)

    stats = {
        "vmin": round(vmin, 6),
        "vmax": round(vmax, 6),
        "vmean": round(sum(vals) / len(vals), 6) if vals else 0.0,
        "count": len(tokens),
        "count_with_lp": len(vals),
        "csv_name": os.path.basename(jsonl_path),
        "path": jsonl_path,
        "global_min": round(gmin, 6),
        "global_max": round(gmax, 6),
    }
    return items, stats


def ensure_cached(jsonl_path: str) -> Tuple[List[dict], dict]:
    cached = PAGE_CACHE.get(jsonl_path)
    if cached:
        return cached
    items, stats = build_items_for_file(jsonl_path)
    PAGE_CACHE[jsonl_path] = (items, stats)
    return items, stats


# -------- HTML Template --------
BASE_TEMPLATE = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Token Logprob Heatmap</title>
  <script src="https://d3js.org/d3.v7.min.js"></script>
  <style>
    body { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace; padding: 20px 30px; line-height: 1.7; padding-bottom: 400px; }
    header { display:flex; align-items:baseline; gap:12px; flex-wrap:wrap; }
    h1 { font-size: 18px; margin: 0 0 6px 0; }
    .meta { font-size: 13px; color: #555; }
    .legend { display: inline-flex; align-items: center; gap: 6px; margin-left: 8px; font-size: 12px; }
    .legend-bar { height: 10px; width: 200px; background: linear-gradient(90deg, hsl(0,85%,70%), hsl(120,85%,70%)); border-radius: 6px; border: 1px solid rgba(0,0,0,0.1); display:inline-block; vertical-align:middle; }
    nav .btn { display:inline-block; padding:4px 8px; border:1px solid #ddd; border-radius:8px; text-decoration:none; color:#111; background:#fafafa; font-size:12px; }
    nav .btn[disabled] { opacity: .5; pointer-events:none; }
    .jump { display:inline-flex; gap:6px; align-items:center; margin-left:10px; }
    .jump input { width:64px; padding:2px 6px; border:1px solid #ddd; border-radius:6px; font-family:inherit; font-size:12px; }
    .jump button { padding:4px 8px; border:1px solid #ddd; border-radius:8px; background:#f7f7f7; cursor:pointer; font-size:12px; }
    .filename { padding:2px 8px; border:1px solid #ddd; border-radius:999px; background:#f7f7f7; }
    .container { white-space: normal; word-break: break-word; max-width: 1200px; font-size: 15px; margin-top: 8px; }
    .token { display: inline; padding: 0 2px; border-radius: 4px; transition: box-shadow 0.12s ease-in-out; }
    .token:hover { box-shadow: 0 0 0 1px rgba(0,0,0,0.18) inset; }
    .tooltip {
      position: fixed;
      z-index: 9999;
      background: #fff;
      border: 1px solid rgba(0,0,0,0.12);
      border-radius: 8px;
      box-shadow: 0 8px 24px rgba(0,0,0,0.12);
      padding: 12px;                 /* 24px total horizontal padding */
      font-size: 12px;
      pointer-events: none;
      width: auto;                    /* let JS set exact width */
      max-width: none;                /* prevent clipping */
      display: none;
    }
    .tooltip svg text { font-size: 10px; fill: #333; }
  </style>
</head>
<body>
  <header>
    <h1>Token Logprob Heatmap</h1>
    <span class="filename">{{ stats.csv_name }}</span>
    <span class="meta">
      Tokens: {{ stats.count }} (with logprob: {{ stats.count_with_lp }})
      · Range: [{{ stats.vmin }}, {{ stats.vmax }}]
      · Mean: {{ stats.vmean }}
      <span class="legend"><span>Low</span><span class="legend-bar"></span><span>High</span></span>
    </span>
  </header>

  {% if multi %}
  <nav style="margin:8px 0 4px 0;">
    <a class="btn" href="{{ prev_url }}" {% if not has_prev %}disabled{% endif %}>← Prev</a>
    <a class="btn" href="{{ next_url }}" {% if not has_next %}disabled{% endif %}>Next →</a>
    <span class="meta" style="margin-left:8px;">Page {{ page_index + 1 }} / {{ total_pages }}</span>

    <!-- Jump to page (1-based) -->
    <span class="jump" aria-label="Jump to page">
      <label for="jumpInput" class="meta">Jump:</label>
      <input id="jumpInput" type="number" min="1" max="{{ total_pages }}" value="{{ page_index + 1 }}" />
      <button id="jumpBtn" type="button">Go</button>
    </span>
  </nav>
  {% endif %}

  <div class="container">{% for item in items -%}
{%- if item.type == 'br' -%}<br>
{%- else -%}
<span class="token"
      style="background-color: {{ item.color }};"
      data-alts='{{ item.alts | tojson }}'
      data-token="{{ item.text }}">{{ item.text }}</span>
{%- endif -%}
{%- endfor %}</div>

  <div id="tooltip" class="tooltip"></div>

  <script>
    const tooltip = document.getElementById('tooltip');
    const OFFSET = 12;
    const globalMin = {{ stats.global_min }};
    const globalMax = {{ stats.global_max }};
    const IS_MULTI = {{ 'true' if multi else 'false' }};
    const TOTAL_PAGES = {{ total_pages }};
    const CUR_INDEX = {{ page_index }}; // zero-based

    // Graph sizing (SVG)
    const SVG_WIDTH  = 460; // inner drawing area width
    const SVG_HEIGHT_MAX = 360;
    const PADDING_X = 24;   // must match .tooltip horizontal padding (12px * 2)

    // Clamp tooltip to viewport so it never overflows to the right
    function placeTooltip(x, y, totalWidth) {
      const vw = window.innerWidth || document.documentElement.clientWidth;
      let left = x + OFFSET;
      if (left + totalWidth + 8 > vw) { // 8px safety
        left = Math.max(8, vw - totalWidth - 8);
      }
      tooltip.style.left = left + 'px';
      tooltip.style.top  = (y + OFFSET) + 'px';
    }

    document.addEventListener('mousemove', (e) => {
      const t = e.target;
      if (t && t.classList && t.classList.contains('token')) {
        const data = JSON.parse(t.getAttribute('data-alts') || '[]');
        const sampled = t.getAttribute('data-token');
        showTooltip(data, sampled, e.clientX, e.clientY);
      } else {
        tooltip.style.display = 'none';
      }
    });

    function showTooltip(data, sampled, x, y) {
      tooltip.style.display = 'block';
      tooltip.innerHTML  = '';
      if (!data.length) { tooltip.textContent = '(no data)'; return; }

      // Compute height based on rows; keep within max
      const height = Math.min(SVG_HEIGHT_MAX, data.length * 14 + 60);
      const margin = { top: 10, right: 40, bottom: 24, left: 130 }; // extra right space for labels
      const totalWidth = SVG_WIDTH + PADDING_X; // content width + padding
      tooltip.style.width = totalWidth + 'px';

      // Place after we know width
      placeTooltip(x, y, totalWidth);

      const svg = d3.select(tooltip).append('svg')
        .attr('width', SVG_WIDTH)
        .attr('height', height);

      data.sort((a, b) => b.logprob - a.logprob);

      const xScale = d3.scaleLinear()
        .domain([globalMin, globalMax])
        .range([margin.left, SVG_WIDTH - margin.right]);

      const yScale = d3.scaleBand()
        .domain(data.map(d => d.decoded_token))
        .range([margin.top, height - margin.bottom])
        .padding(0.1);

      // Bars
      svg.selectAll('rect')
        .data(data)
        .enter()
        .append('rect')
        .attr('x', xScale(globalMin))
        .attr('y', d => yScale(d.decoded_token))
        .attr('width', d => Math.max(1, xScale(d.logprob) - xScale(globalMin)))
        .attr('height', yScale.bandwidth())
        .attr('fill', d => d.decoded_token === sampled ? '#2aa198' : '#93a1a1');

      // Value labels (INSIDE, right-aligned)
      const minX = xScale(globalMin) + 4; // keep text inside even for short bars
      svg.selectAll('text.value')
        .data(data)
        .enter()
        .append('text')
        .attr('class', 'value')
        .attr('x', d => Math.max(xScale(d.logprob) - 4, minX))
        .attr('y', d => yScale(d.decoded_token) + yScale.bandwidth() / 2)
        .attr('dy', '0.32em')
        .attr('font-size', '10px')
        .attr('text-anchor', 'end')
        .attr('fill', '#fff')
        .style('paint-order', 'stroke')
        .style('stroke', 'rgba(0,0,0,0.35)')
        .style('stroke-width', '0.75px')
        .text(d => d.logprob.toFixed(4));

      // Axes
      svg.append('g')
        .attr('transform', `translate(0,${height - margin.bottom})`)
        .call(d3.axisBottom(xScale).ticks(5).tickSizeOuter(0));

      svg.append('g')
        .attr('transform', `translate(${margin.left},0)`)
        .call(d3.axisLeft(yScale).tickSizeOuter(0));
    }

    // --- Keyboard navigation ---
    document.addEventListener('keydown', (e) => {
      if (['INPUT', 'TEXTAREA'].includes(document.activeElement.tagName)) return;
      if (e.key === 'ArrowLeft' && "{{ has_prev }}".toLowerCase() === "true") {
        window.location.href = "{{ prev_url }}";
      } else if (e.key === 'ArrowRight' && "{{ has_next }}".toLowerCase() === "true") {
        window.location.href = "{{ next_url }}";
      }
    });

    // --- Jump to page (1-based) ---
    if (IS_MULTI) {
      const input = document.getElementById('jumpInput');
      const btn = document.getElementById('jumpBtn');

      function gotoPage() {
        const val = parseInt(input.value, 10);
        if (isNaN(val)) return;
        const clamped = Math.max(1, Math.min(TOTAL_PAGES, val));
        // Convert 1-based -> 0-based for route
        const idx = clamped - 1;
        if (idx !== CUR_INDEX) {
          window.location.href = `/file/${idx}`;
        }
      }

      btn?.addEventListener('click', gotoPage);
      input?.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
          gotoPage();
        }
      });
    }
  </script>
</body>
</html>
"""


# -------- Flask routes --------
@app.route("/")
def index():
    if INPUT_IS_DIR:
        if not FILE_LIST:
            abort(404, "No JSONL files found.")
        return redirect(url_for("view_file", idx=0))
    path = FILE_LIST[0]
    items, stats = ensure_cached(path)
    return render_template_string(BASE_TEMPLATE, items=items, stats=stats, multi=False, page_index=0, total_pages=1, has_prev=False, has_next=False, prev_url="#", next_url="#")


@app.route("/file/<int:idx>")
def view_file(idx: int):
    if not INPUT_IS_DIR:
        return redirect(url_for("index"))
    if idx < 0 or idx >= len(FILE_LIST):
        abort(404, "Page index out of range.")
    path = FILE_LIST[idx]
    items, stats = ensure_cached(path)

    has_prev = idx > 0
    has_next = idx + 1 < len(FILE_LIST)
    return render_template_string(
        BASE_TEMPLATE,
        items=items,
        stats=stats,
        multi=True,
        page_index=idx,
        total_pages=len(FILE_LIST),
        has_prev=has_prev,
        has_next=has_next,
        prev_url=url_for("view_file", idx=idx - 1) if has_prev else "#",
        next_url=url_for("view_file", idx=idx + 1) if has_next else "#",
    )


def create_app(input_path: str) -> Flask:
    global INPUT_IS_DIR, FILE_LIST
    if os.path.isdir(input_path):
        INPUT_IS_DIR = True
        FILE_LIST = sorted([os.path.join(input_path, f) for f in os.listdir(input_path) if f.lower().endswith(".jsonl")])
        if not FILE_LIST:
            raise FileNotFoundError("No .jsonl files found.")
    else:
        INPUT_IS_DIR = False
        if not os.path.exists(input_path):
            raise FileNotFoundError(input_path)
        FILE_LIST = [input_path]
    return app


def main():
    parser = argparse.ArgumentParser(description="Token Logprob Heatmap")
    parser.add_argument("input_path", help="Path to JSONL file or folder")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()
    create_app(args.input_path)
    app.run(host=args.host, port=args.port, debug=True)


if __name__ == "__main__":
    main()
