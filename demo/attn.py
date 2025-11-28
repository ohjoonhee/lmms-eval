#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AttentionViz — BertViz‑style, single‑file Flask app (UPDATED for per‑step attention)

New in this version
- Supports attention saved as: List[ num_generated_tokens ] of List[ num_layers ] of Tensor[B, H, S, S]
  (also accepts [H,S,S] if batch has been sliced already).
- UI selector for generation step (token index) to visualize attention at that step.
- Each step may have a different sequence length S; the viewer adapts tokens & grid size.

Other features
- Loads decoded tokens from a *.pt file (list[str] or dict with common keys)
- Endpoints: /meta?gen=, /attn?gen=&layer=&head=
- Options: head‑avg toggle, cell size control, hover highlight

Usage
    python attention_viz_app.py \
        --attn-path path/to/attentions.pt \
        --tokens-path path/to/tokens.pt \
        --batch-idx 0 \
        --host 127.0.0.1 --port 5000 --debug

Inputs
- Attention: List[T][L][B,H,S,S] (preferred), also supports [T][L][H,S,S]
- Tokens: list[str] or dict with key among {'tokens','input_tokens','decoded','decoded_tokens'}

Dependencies
    pip install flask torch
"""

from __future__ import annotations
import argparse
import os
from typing import Any, Dict, List, Optional

from flask import Flask, jsonify, request, render_template_string, abort, send_from_directory

try:
    import torch
except Exception as e:
    raise SystemExit("PyTorch is required: pip install torch " + str(e))

app = Flask(__name__)

# -----------------------
# Globals
# -----------------------
ATTN_STEPS: List[torch.Tensor] = []  # each: [L,H,S,S]
TOKENS_ALL: List[str] = []  # full token list (we slice per step)
META0: Dict[str, int] = {"layers": 0, "heads": 0}

# -----------------------
# Utilities
# -----------------------


def _to_list(x: torch.Tensor) -> List[List[float]]:
    return x.detach().cpu().tolist()


def _extract_steps(obj: Any, batch_idx: Optional[int]) -> List[torch.Tensor]:
    """Extract a list of [L,H,S,S] tensors from supported formats.

    Expected primary format: List[T][L][B,H,S,S]. We also accept [T][L][H,S,S].
    """
    # Unwrap dicts with common keys
    if isinstance(obj, dict):
        for k in ["attn", "attns", "attentions", "attention", "all_attentions", "hidden_attentions"]:
            if k in obj:
                obj = obj[k]
                break

    if not isinstance(obj, (list, tuple)):
        raise ValueError("Expected a list/tuple for per‑step attentions (List[T][L][B,H,S,S]).")

    T = len(obj)
    if T == 0:
        raise ValueError("Empty attention list (T=0)")

    steps: List[torch.Tensor] = []
    for t_idx in range(T):
        layer_list = obj[t_idx]
        if not isinstance(layer_list, (list, tuple)):
            raise ValueError(f"Step {t_idx}: expected List[L][...], got {type(layer_list)}")
        layers: List[torch.Tensor] = []
        for li, layer_t in enumerate(layer_list):
            t = torch.as_tensor(layer_t)
            if t.dim() == 4:  # [B,H,S,S]
                b = 0 if batch_idx is None else int(batch_idx)
                if b >= t.size(0):
                    raise IndexError(f"batch_idx {b} out of range for layer {li} with B={t.size(0)}")
                t = t[b]  # -> [H,S,S]
            elif t.dim() == 3:  # [H,S,S]
                pass
            else:
                raise ValueError(f"Step {t_idx} layer {li}: expected 3D/4D tensor, got shape {tuple(t.shape)}")
            layers.append(t)
        step_tensor = torch.stack(layers, dim=0)  # [L,H,S,S]
        steps.append(step_tensor)
    return steps


def _load_tokens(obj: Any) -> List[str]:
    if isinstance(obj, dict):
        for k in ["tokens", "input_tokens", "decoded", "decoded_tokens", "ids2str", "words"]:
            if k in obj:
                obj = obj[k]
                break
    if isinstance(obj, torch.Tensor):
        if obj.dtype in (torch.int16, torch.int32, torch.int64):
            return [str(int(i)) for i in obj.flatten().tolist()]
        obj = obj.tolist()
    if isinstance(obj, (list, tuple)):
        if len(obj) > 0 and isinstance(obj[0], (list, tuple)) and all(isinstance(x, str) for x in obj[0]):
            obj = obj[0]
        return [str(x) for x in obj]
    if isinstance(obj, str):
        return obj.split()
    raise ValueError("Unsupported token container. Provide a list[str] or dict with 'tokens'.")


# -----------------------
# Routes
# -----------------------
@app.route("/")
def index():
    return render_template_string(INDEX_HTML)


@app.route("/meta")
def meta():
    if not ATTN_STEPS:
        abort(500, description="Attention not loaded")
    gen = int(request.args.get("gen", 0))
    gen = max(0, min(gen, len(ATTN_STEPS) - 1))
    attn = ATTN_STEPS[gen]
    L, H, S, S2 = attn.shape
    assert S == S2
    tokens = TOKENS_ALL[:S] if TOKENS_ALL else [f"t{i}" for i in range(S)]
    if len(tokens) < S:
        tokens = tokens + ["∅"] * (S - len(tokens))
    return jsonify(
        {
            "gen_len": len(ATTN_STEPS),
            "gen_idx": gen,
            "layers": L,
            "heads": H,
            "seq_len": S,
            "tokens": tokens,
        }
    )


@app.route("/attn")
def attn_slice():
    if not ATTN_STEPS:
        abort(500, description="Attention not loaded")
    try:
        gen = int(request.args.get("gen", 0))
        gen = max(0, min(gen, len(ATTN_STEPS) - 1))
        attn = ATTN_STEPS[gen]
        layer = int(request.args.get("layer", 0))
        head = request.args.get("head", "avg")
        if head == "avg":
            mat = attn[layer].mean(dim=0)  # [S,S]
        else:
            head = int(head)
            mat = attn[layer, head]
        mat = torch.softmax(mat, dim=-1)
        return jsonify(
            {
                "gen": gen,
                "layer": layer,
                "head": head,
                "matrix": _to_list(mat),
            }
        )
    except Exception as e:
        abort(400, description=str(e))


@app.route("/attn_grid")
def attn_grid():
    """Return all heads for all layers for a selected generation step.
    Response shape: { matrices: List[L][H][S][S] }
    Note: this can be large; intended for small/medium S.
    Optional query: topk (int) — applied client-side; we still send full matrix.
    """
    if not ATTN_STEPS:
        abort(500, description="Attention not loaded")
    gen = int(request.args.get("gen", 0))
    gen = max(0, min(gen, len(ATTN_STEPS) - 1))
    attn = ATTN_STEPS[gen]
    L, H, S, _ = attn.shape
    # softmax normalize per (q, head, layer)
    mats: List[List[List[List[float]]]] = []
    for l in range(L):
        row = []
        for h in range(H):
            mat = torch.softmax(attn[l, h], dim=-1)
            row.append(_to_list(mat))
        mats.append(row)
    return jsonify({"gen": gen, "layers": L, "heads": H, "seq_len": S, "matrices": mats})


# -----------------------
# HTML/JS (inline, Attention Grid Style)
# -----------------------
INDEX_HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>AttentionViz — BertViz‑style (Grid)</title>
  <style>
    :root { --bg:#0b0b0f; --fg:#eaeaf2; --muted:#9aa0a6; --grid:#2a2a35; --accent:#8ab4f8; }
    body { margin:0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, Apple Color Emoji, Segoe UI Emoji; background:var(--bg); color:var(--fg); }
    header { display:flex; gap:1rem; align-items:center; padding:12px 16px; border-bottom:1px solid var(--grid); position:sticky; top:0; background:var(--bg); z-index:10; flex-wrap: wrap; }
    select, input { background:#111217; color:var(--fg); border:1px solid var(--grid); border-radius:8px; padding:6px 10px; }
    .grid { padding:16px; display:grid; gap:10px; }
    .cell { background:#0a0a0d; border:1px solid #141421; border-radius:8px; padding:6px; position:relative; }
    .cell canvas { display:block; width:100%; height:100%; }
    .rowlbl, .collbl { color:#cfd3dc; font-weight:600; }
    .rowlbl { writing-mode: vertical-rl; text-orientation: mixed; }
    .tokens { padding:0 16px 12px 16px; color:var(--muted); font-size:12px; }
    .panel { padding:8px 16px; display:flex; gap:12px; align-items:center; color:var(--muted); }
    .panel .spacer { flex:1; }
  </style>
</head>
<body>
  <header>
    <strong>AttentionViz — Grid</strong>
    <label>Gen step <select id="gen"></select></label>
    <label>Top‑K per query <input id="topk" type="number" value="4" min="1" max="32" step="1" style="width:64px"/></label>
    <label>Canvas px (cell) <input id="cellpx" type="number" value="140" min="80" max="260" step="10" style="width:72px"/></label>
    <span id="shape" style="color:var(--muted)"></span>
  </header>
  <div class="panel">
    <div>Each cell = one head. Rows = layers, Cols = heads. Lines show strongest attention per query (Top‑K). Colors vary by layer.</div>
    <div class="spacer"></div>
  </div>
  <div class="tokens" id="tokens"></div>
  <div id="grid" class="grid"></div>
  <script>
    const $ = (s)=>document.querySelector(s);
    const grid = $('#grid');
    const genSel = $('#gen');
    const topkInp = $('#topk');
    const cellPxInp = $('#cellpx');
    const tokensEl = $('#tokens');
    const shapeEl = $('#shape');

    // Color palette per layer (cycle if more layers)
    const layerColors = [ '#75a7ff', '#ff9f50', '#63d97e', '#ff6a63', '#b38cff', '#7dd6d0', '#f7d774' ];

    function makeGrid(L, H, cellPx){
      grid.style.gridTemplateColumns = `repeat(${H}, ${cellPx}px)`;
      grid.style.gridTemplateRows = `repeat(${L}, ${cellPx}px)`;
      grid.innerHTML = '';
      const cells = [];
      for(let l=0;l<L;l++){
        for(let h=0;h<H;h++){
          const div = document.createElement('div');
          div.className = 'cell';
          const cv = document.createElement('canvas');
          cv.width = cellPx; cv.height = cellPx;
          div.appendChild(cv);
          grid.appendChild(div);
          cells.push({l,h,cv});
        }
      }
      return cells;
    }

    function drawCell(canvas, mat, color, topK){
      const ctx = canvas.getContext('2d');
      const W = canvas.width, Hh = canvas.height;
      ctx.clearRect(0,0,W,Hh);
      const S = mat.length;
      if(S === 0) return;
      const margin = 8;
      const xL = margin, xR = W - margin;
      const stepY = (Hh - 2*margin) / Math.max(1, S-1);
      ctx.lineWidth = 1;
      for(let q=0; q<S; q++){
        const yq = margin + q * stepY;
        const row = mat[q];
        // find topK indices
        const idxs = Array.from({length: row.length}, (_,i)=>i)
                          .sort((a,b)=> row[b]-row[a])
                          .slice(0, topK);
        for(const k of idxs){
          const w = row[k];
          const yk = margin + k * stepY;
          ctx.beginPath();
          // alpha scaled by sqrt(weight) for dynamic range
          const alpha = Math.min(1, Math.sqrt(Math.max(0, w)));
          ctx.strokeStyle = hexWithAlpha(color, alpha*0.9);
          ctx.moveTo(xL, yq);
          ctx.lineTo(xR, yk);
          ctx.stroke();
        }
      }
      // draw side rails
      ctx.strokeStyle = 'rgba(255,255,255,0.08)';
      ctx.beginPath(); ctx.moveTo(xL, margin); ctx.lineTo(xL, Hh-margin); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(xR, margin); ctx.lineTo(xR, Hh-margin); ctx.stroke();
    }

    function hexWithAlpha(hex, alpha){
      const c = hex.replace('#','');
      const r = parseInt(c.substring(0,2),16);
      const g = parseInt(c.substring(2,4),16);
      const b = parseInt(c.substring(4,6),16);
      return `rgba(${r},${g},${b},${alpha})`;
    }

    async function fetchMeta(gen){
      const res = await fetch(`/meta?gen=${gen}`);
      if(!res.ok){ throw new Error(await res.text()); }
      return await res.json();
    }

    async function fetchGrid(gen){
      const res = await fetch(`/attn_grid?gen=${gen}`);
      if(!res.ok){ throw new Error(await res.text()); }
      return await res.json();
    }

    async function render(){
      const gen = Number(genSel.value);
      const meta = await fetchMeta(gen);
      const data = await fetchGrid(gen);
      const L = data.layers, Hh = data.heads, S = data.seq_len;
      shapeEl.textContent = `Gen ${gen+1}/${meta.gen_len} — L=${L}, H=${Hh}, S=${S}`;
      tokensEl.textContent = meta.tokens.join(' ');
      const cellPx = Number(cellPxInp.value);
      const cells = makeGrid(L, Hh, cellPx);
      const topK = Number(topkInp.value);
      // Draw each cell
      let idx = 0;
      for(let l=0;l<L;l++){
        const color = layerColors[l % layerColors.length];
        for(let h=0;h<Hh;h++){
          const mat = data.matrices[l][h];
          drawCell(cells[idx++].cv, mat, color, topK);
        }
      }
    }

    async function init(){
      const first = await fetchMeta(0);
      genSel.innerHTML = Array.from({length: first.gen_len}, (_,i)=>`<option value="${i}">${i}</option>`).join('');
      await render();
    }

    genSel.onchange = ()=>render();
    topkInp.onchange = ()=>render();
    cellPxInp.onchange = ()=>render();

    init();
  </script>
</body>
</html>
"""

# -----------------------
# CLI & bootstrap
# -----------------------


def main():
    parser = argparse.ArgumentParser(description="AttentionViz — visualize attentions like BertViz (per‑step format)")
    parser.add_argument("--attn-path", required=True, help="Path to torch.save'd attentions (*.pt) — List[T][L][B,H,S,S]")
    parser.add_argument("--tokens-path", required=True, help="Path to torch.save'd decoded tokens (*.pt)")
    parser.add_argument("--batch-idx", type=int, default=None, help="Batch index for [B,H,S,S] tensors")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.attn_path):
        raise SystemExit(f"Not found: {args.attn_path}")
    if not os.path.exists(args.tokens_path):
        raise SystemExit(f"Not found: {args.tokens_path}")

    attn_obj = torch.load(args.attn_path, map_location="cpu")
    tokens_obj = torch.load(args.tokens_path, map_location="cpu")

    global ATTN_STEPS, TOKENS_ALL, META0
    ATTN_STEPS = _extract_steps(attn_obj, args.batch_idx)  # list of [L,H,S,S]
    TOKENS_ALL = _load_tokens(tokens_obj)

    if not ATTN_STEPS:
        raise SystemExit("No attention steps extracted")

    L, H, S, _ = ATTN_STEPS[0].shape
    META0 = {"layers": L, "heads": H}
    print(f"Loaded {len(ATTN_STEPS)} steps; first step: L={L}, H={H}, S={S}; tokens={len(TOKENS_ALL)}")

    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
