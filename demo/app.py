#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, json, argparse, pathlib, math, io, re, base64
from datetime import datetime

from flask import Flask, request, abort, send_file, redirect, url_for, render_template_string, make_response
from markupsafe import Markup, escape  # Flask 3.x: import from markupsafe

# Optional deps: datasets / pillow / numpy (install if needed)
HF_AVAILABLE = False
try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except Exception:
    load_dataset = None
try:
    import numpy as np
except Exception:
    np = None

from PIL import Image

app = Flask(__name__)

# -------- Settings / Utils --------
DATA_PATH = None
DEFAULT_PAGE_SIZE = 1  # default items per page
MAX_PAGE_SIZE = 500

# Hugging Face dataset globals
HF_DATASET = None  # datasets.Dataset
HF_IMAGE_COL = None  # str
HF_LENGTH = 0  # int

# In-memory thumbnail cache: { doc_id: (bytes, mimetype) }
THUMB_CACHE = {}
THUMB_MAX = 512  # max side in pixels (configurable via argparse)


def ensure_file(path):
    if not path or not os.path.isfile(path):
        abort(404, description="JSONL file not found.")


def iter_jsonl(path):
    """
    Read JSONL line-by-line. Yield both parse-success and parse-failure lines.
    Empty lines are skipped. We attach a 0-based 'recno' to align with the HF dataset index.
    NOTE: 'recno' is only the file-order index; image lookup now uses 'doc_id' from the item itself.
    """
    recno = 0
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            raw = line.rstrip("\n")
            if not raw.strip():
                continue
            try:
                obj = json.loads(raw)
                yield {"lineno": lineno, "recno": recno, "ok": True, "data": obj, "raw": raw}
            except Exception as e:
                yield {"lineno": lineno, "recno": recno, "ok": False, "data": None, "raw": raw, "error": f"{type(e).__name__}: {e}"}
            recno += 1


_cache = {"count": None, "mtime": None}


def file_meta(path):
    p = pathlib.Path(path)
    stat = p.stat()
    return {"size": stat.st_size, "mtime": stat.st_mtime, "name": p.name, "path": str(p.resolve())}


def count_lines_cached(path):
    mtime = os.path.getmtime(path)
    if _cache["count"] is not None and _cache["mtime"] == mtime:
        return _cache["count"]
    c = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                c += 1
    _cache["count"] = c
    _cache["mtime"] = mtime
    return c


def matches_query(obj, raw, query):
    if not query:
        return True
    q = query.lower()
    try:
        if isinstance(obj, (dict, list)):
            text = json.dumps(obj, ensure_ascii=False)
        elif obj is None:
            text = "null"
        else:
            text = str(obj)
    except Exception:
        text = raw
    return q in text.lower()


def get_sort_keyfunc(sort_key):
    if not sort_key:
        return None

    def _key(item):
        data = item["data"]
        if isinstance(data, dict) and sort_key in data:
            v = data.get(sort_key)
            t = 0
            if isinstance(v, (int, float)):
                t = 0
            elif v is None:
                t = 2
            else:
                t = 1
            return (t, v)
        return (3, None)

    return _key


def human_size(n):
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024.0:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PB"


def human_time(ts):
    try:
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(ts)


def build_pages(curr, total_pages, window=3):
    pages = set([1, total_pages, curr])
    for d in range(1, window + 1):
        pages.add(max(1, curr - d))
        pages.add(min(total_pages, curr + d))
    return sorted(pages)


def key_preview_html(data):
    if not isinstance(data, dict):
        return ""
    keys = list(data.keys())[:3]
    parts = []
    for k in keys:
        v = data.get(k)
        vv = "[…]" if isinstance(v, (dict, list)) else v
        parts.append(f"<code class='kv'>{k}</code>=<span class='muted'>{vv}</span>")
    return " • ".join(parts)


def build_chips(data, max_len=40):
    """Build short summary chips for common keys."""
    if not isinstance(data, dict):
        return []
    keys = ["id", "doc_id", "image_id", "filename", "file", "path", "label", "category", "class", "prompt", "question", "answer", "caption", "text"]
    chips = []
    for k in keys:
        if k in data:
            v = data.get(k)
            if isinstance(v, (dict, list)):
                continue
            s = str(v)
            if len(s) > max_len:
                s = s[: max_len - 1] + "…"
            chips.append({"k": k, "v": s})
    return chips[:6]


def highlight_text(s: str, q: str):
    if not s:
        return None
    if not q:
        return Markup(escape(s))
    esc = str(escape(s))
    pattern = re.compile(re.escape(q), re.IGNORECASE)
    marked = pattern.sub(lambda m: f"<mark>{m.group(0)}</mark>", esc)
    return Markup(marked)


def common_query_args():
    q = request.args.get("q") or ""
    sort_key = request.args.get("sort_key") or ""
    page_size = min(max(int(request.args.get("page_size", DEFAULT_PAGE_SIZE)), 1), MAX_PAGE_SIZE)
    compact = request.args.get("compact", "1") != "0"  # default ON
    return q, sort_key, page_size, compact


# -------- Templates (updated to use doc_id) --------
PAGE_TMPL = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{{ meta.name }} — JSONL Viewer</title>
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <style>
    :root { --bg:#0b0c10; --card:#121318; --fg:#e8e8e8; --muted:#9aa0a6; --accent:#5aa9e6; }
    * { box-sizing: border-box; }
    body { margin:0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, Apple Color Emoji, Segoe UI Emoji; background: var(--bg); color: var(--fg); }
    header { padding: 20px; border-bottom: 1px solid #222; display:flex; gap:16px; align-items: baseline; flex-wrap: wrap; }
    h1 { margin:0; font-size: 1.25rem; display:flex; align-items:center; gap:8px; }
    .muted { color: var(--muted); font-size: .9rem; }
    main { padding: 16px; max-width: 1200px; margin: 0 auto; }
    form.filters { display:flex; flex-wrap: wrap; gap:12px; align-items: flex-end; margin-bottom: 16px; }
    label { display:flex; flex-direction:column; gap:4px; }
    input, select { background: #0f1116; border: 1px solid #2a2d34; color: var(--fg); padding: 8px 10px; border-radius: 8px; }
    button { padding: 8px 12px; border-radius: 8px; background: var(--accent); border: none; color: #012a4a; font-weight: 700; cursor: pointer; }
    .card { background: var(--card); border: 1px solid #1c1f26; border-radius: 12px; padding: 12px; margin-bottom: 12px; }
    .rowhead { display:flex; justify-content: space-between; gap: 8px; align-items: baseline; }
    .lineno { color: var(--muted); font-size: .9rem; }
    pre { margin: 8px 0 0; overflow: auto; white-space: pre-wrap; word-break: break-word; background: #0f1116; padding: 10px; border-radius: 8px; border:1px solid #22252c; }
    .bad { border-color: #6b1b1b; }
    .pill { display:inline-block; font-size: .75rem; padding: 2px 8px; border-radius: 999px; border: 1px solid #2a2d34; color: var(--muted); }
    nav.pages { display:flex; gap:6px; flex-wrap: wrap; margin-top: 16px; }
    nav.pages a, nav.pages span { padding:6px 10px; border-radius: 8px; border:1px solid #2a2d34; text-decoration:none; color: var(--fg); }
    nav.pages .current { background:#1b1f26; }
    .toolbar { display:flex; gap:10px; align-items:center; flex-wrap:wrap; }
    .link { color: var(--accent); text-decoration: none; }
    .flex { display:flex; gap:10px; align-items:center; flex-wrap:wrap; }
    .grow { flex:1 }
    code.kv { background:#0f1116; border:1px solid #22252c; padding:2px 6px; border-radius:6px; }
    .section-title { margin-top: 8px; font-weight:700; font-size:.9rem; color:#cfd5db; }
    .imgbox { margin-top: 8px; border:1px solid #22252c; border-radius:10px; overflow:hidden; background:#0f1116; }
    .imgbox img { display:block; width: 100%; height: auto; }
    .imgcap { padding:6px 10px; color: #9aa0a6; font-size:.85rem; border-top:1px solid #22252c; }
    .thumb { max-width: 480px; }
    .chips { display:flex; flex-wrap:wrap; gap:6px; margin-top:6px }
    .chip { border:1px solid #2a2d34; background:#0f1116; border-radius:999px; padding:2px 8px; font-size:.78rem; color:#c8ccd1 }

    /* Lightbox */
    .lightbox { position: fixed; inset: 0; background: rgba(0,0,0,.85); display: none; align-items: center; justify-content: center; z-index: 9999; }
    .lightbox.open { display: flex; }
    .lightbox img { max-width: 95vw; max-height: 95vh; }

    /* Copy button */
    .iconbtn {
      display:inline-flex; align-items:center; justify-content:center;
      width:24px; height:24px; padding:0; margin-left:4px;
      border-radius:6px; border:1px solid #2a2d34; background:#0f1116;
      color:#cfd5db; cursor:pointer;
    }
    .iconbtn:hover { background:#151822; }
    .iconbtn:active { transform: translateY(1px); }
    .iconbtn:focus { outline:2px solid #5aa9e6; outline-offset:2px; }
    .copy-toast { font-size:.85rem; color:#9aa0a6; display:none }

    /* Copy/Check icon swap */
    .icon-copy { display:inline; }
    .icon-check { display:none; }
    .iconbtn.ok .icon-copy { display:none; }
    .iconbtn.ok .icon-check { display:inline; }
  </style>
  <script>
    function goDetail(idx) { window.location = "/item/" + idx + window.location.search.replace(/^\?/, "&"); }
  </script>
</head>
<body>
<header>
  <h1>
    {{ meta.name }}
    <button class="iconbtn" type="button"
            data-path="{{ meta.path|e }}"
            onclick="copyPath(this.dataset.path, this)"
            aria-label="Copy file path">
      <!-- copy icon -->
      <svg class="icon-copy" width="16" height="16" viewBox="0 0 24 24" fill="none"
           xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
        <path d="M8 8H18V18H8V8Z" stroke="currentColor" stroke-width="2"/>
        <path d="M6 16H5C4.44772 16 4 15.5523 4 15V5C4 4.44772 4.44772 4 5 4H15C15.5523 4 16 4.44772 16 5V6"
              stroke="currentColor" stroke-width="2"/>
      </svg>
      <!-- check icon -->
      <svg class="icon-check" width="16" height="16" viewBox="0 0 24 24" fill="none"
           xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
        <path d="M20 6L9 17L4 12" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
      </svg>
    </button>
    <span class="copy-toast" role="status" aria-live="polite">Copied!</span>
  </h1>
  <div class="muted">Total {{ total }} lines • {{ size_human }} • Last modified {{ mtime_human }}</div>
</header>

<main>
  <form class="filters" method="get" action="/">
    <label style="min-width:260px" title="Substring search across record text">
      <span class="muted">Search (substring)</span>
      <input class="grow" type="search" name="q" value="{{ q or '' }}" placeholder="e.g., cat, id:123" />
    </label>

    <label style="width:120px" title="Page number (starts at 1)">
      <span class="muted">Page</span>
      <input type="number" name="page" min="1" value="{{ page }}" />
    </label>

    <label style="width:140px" title="Items per page">
      <span class="muted">Page size</span>
      <input type="number" name="page_size" min="1" max="{{ max_page_size }}" value="{{ page_size }}" />
    </label>

    <label style="min-width:180px" title="Sort by a top-level key">
      <span class="muted">Sort key (optional)</span>
      <input type="text" name="sort_key" value="{{ sort_key or '' }}" placeholder="e.g., id, ts" />
    </label>

    <input type="hidden" name="compact" value="{{ 1 if compact else 0 }}">
    <button type="submit">Apply</button>

    <a class="link" href="{{ url_for('index', page=page, q=q or None, sort_key=sort_key or None, page_size=page_size if page_size != 1 else None, compact=(0 if compact else 1)) }}">
      View: {{ 'Compact ON' if compact else 'Compact OFF' }}
    </a>
    <a class="link" href="{{ url_for('download') }}">Download original</a>
  </form>

  {% if sort_key %}
  <div class="muted" style="margin-bottom:8px">Sorted by: <code class="kv">{{ sort_key }}</code></div>
  {% endif %}

  {% for item in items %}
    <div class="card {% if not item.ok %}bad{% endif %}">
      <div class="rowhead">
        <div class="flex">
          <span class="pill">#{{ item.index }}</span>
          <span class="lineno">line {{ item.lineno }}</span>
          {% if item.ok and item.key_preview %}
            <span class="lineno">• {{ item.key_preview|safe }}</span>
          {% endif %}
        </div>
        <div class="toolbar">
          <a class="link" href="javascript:void(0)" onclick="goDetail({{ item.index }})">Details</a>
        </div>
      </div>

      {% if item.chips %}
        <div class="chips">
          {% for c in item.chips %}
            <span class="chip"><strong style="opacity:.9">{{ c.k }}</strong>: {{ c.v }}</span>
          {% endfor %}
        </div>
      {% endif %}

      {% if item.has_image %}
        <div class="section-title">Image (click to enlarge)</div>
        <div class="imgbox thumb">
          <a href="{{ url_for('serve_image', doc_id=item.doc_id) }}" class="lb" data-src="{{ url_for('serve_image', doc_id=item.doc_id) }}">
            <img src="{{ url_for('serve_thumb', doc_id=item.doc_id) }}" alt="image for doc_id {{ item.doc_id }}" loading="lazy">
          </a>
          <div class="imgcap">dataset[{{ item.doc_id }}].{{ image_col }} • thumbnail (max {{ thumb_max }}px)</div>
        </div>
      {% endif %}

      {% if item.ok %}
        {% if item.resp_text %}
          <div class="section-title">filtered_resps[0]</div>
          <pre>{{ item.resp_marked or item.resp_text }}</pre>
        {% endif %}
        <details {% if compact %}open{% endif %} style="margin-top:8px">
          <summary class="section-title" style="cursor:pointer">Original JSON</summary>
          <pre>{{ item.pretty }}</pre>
        </details>
      {% else %}
        <pre>⚠️ JSON parse failed: {{ item.error }}
Raw: {{ item.raw }}</pre>
      {% endif %}
    </div>
  {% endfor %}

  <nav class="pages">
    {% for p in pages %}
      {% if p == page %}
        <span class="current">{{ p }}</span>
      {% else %}
        <a href="{{ url_for('index', **page_link_args(p)) }}"
           {% if p == page-1 %}rel="prev"{% elif p == page+1 %}rel="next"{% endif %}>{{ p }}</a>
      {% endif %}
    {% endfor %}
  </nav>
</main>

<!-- Lightbox -->
<div id="lightbox" class="lightbox" onclick="this.classList.remove('open')">
  <img id="lightbox-img" src="" alt="">
</div>
<script>
  // Lightbox behavior
  document.addEventListener('click', (e) => {
    const a = e.target.closest('a.lb');
    if (!a) return;
    e.preventDefault();
    const lb = document.getElementById('lightbox');
    const img = document.getElementById('lightbox-img');
    img.src = a.dataset.src || a.href;
    lb.classList.add('open');
  });

  // Keyboard: ESC to close lightbox, j/→ next page, k/← prev page, / focus search
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      const lb = document.getElementById('lightbox');
      if (lb) lb.classList.remove('open');
      return;
    }

    // If focusing an input/textarea/contenteditable, ignore nav keys (except '/')
    const active = document.activeElement;
    if (active && (active.tagName === 'INPUT' || active.tagName === 'TEXTAREA' || active.isContentEditable)) {
      if (e.key === '/') { e.preventDefault(); }
      return;
    }

    if (e.key === '/') {
      const search = document.querySelector('input[type="search"]');
      if (search) { search.focus(); e.preventDefault(); }
      return;
    }

    if (e.key === 'j' || e.key === 'ArrowRight') {
      const a = document.querySelector('nav.pages a[rel="next"]');
      if (a) { e.preventDefault(); window.location = a.href; }
    } else if (e.key === 'k' || e.key === 'ArrowLeft') {
      const a = document.querySelector('nav.pages a[rel="prev"]');
      if (a) { e.preventDefault(); window.location = a.href; }
    }
  });

  // Copy path to clipboard + toast + icon swap
  async function copyPath(path, btn) {
    try {
      await navigator.clipboard.writeText(path);
    } catch (e) {
      // Fallback for environments without Clipboard API permissions
      const ta = document.createElement('textarea');
      ta.value = path; document.body.appendChild(ta);
      ta.select(); document.execCommand('copy'); document.body.removeChild(ta);
    }
    // Show toast
    const toast = btn.parentElement.querySelector('.copy-toast') || btn.closest('h1')?.querySelector('.copy-toast');
    if (toast) {
      toast.style.display = 'inline';
      setTimeout(() => { toast.style.display = 'none'; }, 1200);
    }
    // Icon swap to check for 1.2s
    btn.classList.add('ok');
    setTimeout(() => { btn.classList.remove('ok'); }, 1200);
  }
</script>
</body>
</html>
"""

DETAIL_TMPL = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>#{{ index }} — {{ meta.name }}</title>
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <style>
    body { margin:0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial; background:#0b0c10; color:#e8e8e8; }
    header { padding: 16px; border-bottom: 1px solid #222; display:flex; gap:12px; align-items:baseline; flex-wrap:wrap; }
    main { padding:16px; max-width: 1000px; margin:0 auto; }
    a { color:#5aa9e6; text-decoration:none; }
    pre { background:#0f1116; border:1px solid #22252c; padding:12px; border-radius:10px; white-space: pre-wrap; word-break: break-word; }
    .muted { color:#9aa0a6 }
    .pill { display:inline-block; font-size:.75rem; padding:2px 8px; border-radius:999px; border:1px solid #2a2d34; color:#9aa0a6; }
    .kv { background:#0f1116; border:1px solid #22252c; padding:2px 6px; border-radius:6px; }
    .section-title { margin-top: 8px; font-weight:700; font-size:.9rem; color:#cfd5db; }
    .imgbox { margin-top: 8px; border:1px solid #22252c; border-radius:10px; overflow:hidden; background:#0f1116; }
    .imgbox img { display:block; width: 100%; height: auto; }
    .imgcap { padding:6px 10px; color:#9aa0a6; font-size:.85rem; border-top:1px solid #22252c; }
    .thumb { max-width: 640px; }

    /* Lightbox */
    .lightbox { position: fixed; inset: 0; background: rgba(0,0,0,.85); display: none; align-items: center; justify-content: center; z-index: 9999; }
    .lightbox.open { display: flex; }
    .lightbox img { max-width: 95vw; max-height: 95vh; }

    /* Copy button */
    .iconbtn {
      display:inline-flex; align-items:center; justify-content:center;
      width:24px; height:24px; padding:0; margin-left:6px;
      border-radius:6px; border:1px solid #2a2d34; background:#0f1116;
      color:#cfd5db; cursor:pointer;
    }
    .iconbtn:hover { background:#151822; }
    .iconbtn:active { transform: translateY(1px); }
    .iconbtn:focus { outline:2px solid #5aa9e6; outline-offset:2px; }
    .copy-wrap { margin-left:auto; display:flex; align-items:center; gap:8px }
    .copy-toast { font-size:.85rem; color:#9aa0a6; display:none }

    /* Copy/Check icon swap */
    .icon-copy { display:inline; }
    .icon-check { display:none; }
    .iconbtn.ok .icon-copy { display:none; }
    .iconbtn.ok .icon-check { display:inline; }
  </style>
</head>
<body>
  <header>
    <a href="{{ url_for('index') }}">← Back to list</a>
    <span class="pill">#{{ index }}</span>
    <span class="muted">line {{ lineno }}</span>
    {% if sort_key %}<span class="muted">• sort: <code class="kv">{{ sort_key }}</code></span>{% endif %}

    <!-- Right side: file name + copy button -->
    <span class="copy-wrap">
      <span class="muted">{{ meta.name }}</span>
      <button class="iconbtn" type="button"
              data-path="{{ meta.path|e }}"
              onclick="copyPath(this.dataset.path, this)"
              aria-label="Copy file path">
        <!-- copy icon -->
        <svg class="icon-copy" width="16" height="16" viewBox="0 0 24 24" fill="none"
             xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
          <path d="M8 8H18V18H8V8Z" stroke="currentColor" stroke-width="2"/>
          <path d="M6 16H5C4.44772 16 4 15.5523 4 15V5C4 4.44772 4.44772 4 5 4H15C15.5523 4 16 4.44772 16 5V6"
                stroke="currentColor" stroke-width="2"/>
        </svg>
        <!-- check icon -->
        <svg class="icon-check" width="16" height="16" viewBox="0 0 24 24" fill="none"
             xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
          <path d="M20 6L9 17L4 12" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
      </button>
      <span class="copy-toast" role="status" aria-live="polite">Copied!</span>
    </span>
  </header>
  <main>
    {% if has_image %}
      <div class="section-title">Image</div>
      <div class="imgbox thumb">
        <a href="{{ url_for('serve_image', doc_id=doc_id) }}" class="lb" data-src="{{ url_for('serve_image', doc_id=doc_id) }}">
          <img src="{{ url_for('serve_thumb', doc_id=doc_id) }}" alt="image for doc_id {{ doc_id }}" loading="lazy">
        </a>
        <div class="imgcap">dataset[{{ doc_id }}].{{ image_col }} • thumbnail (max {{ thumb_max }}px) • click to enlarge</div>
      </div>
    {% endif %}

    {% if ok %}
      {% if resp_text %}
        <div class="section-title">filtered_resps[0]</div>
        <pre>{{ resp_marked or resp_text }}</pre>
      {% endif %}
      <details open style="margin-top:8px">
        <summary class="section-title" style="cursor:pointer">Original JSON</summary>
        <pre>{{ pretty }}</pre>
      </details>
    {% else %}
      <pre>⚠️ JSON parse failed: {{ error }}
Raw: {{ raw }}</pre>
    {% endif %}
  </main>

  <!-- Lightbox -->
  <div id="lightbox" class="lightbox" onclick="this.classList.remove('open')">
    <img id="lightbox-img" src="" alt="">
  </div>
  <script>
    document.addEventListener('click', (e) => {
      const a = e.target.closest('a.lb');
      if (!a) return;
      e.preventDefault();
      const lb = document.getElementById('lightbox');
      const img = document.getElementById('lightbox-img');
      img.src = a.dataset.src || a.href;
      lb.classList.add('open');
    });
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') document.getElementById('lightbox').classList.remove('open');
    });

    // Copy path to clipboard + toast + icon swap
    async function copyPath(path, btn) {
      try {
        await navigator.clipboard.writeText(path);
      } catch (e) {
        const ta = document.createElement('textarea');
        ta.value = path; document.body.appendChild(ta);
        ta.select(); document.execCommand('copy'); document.body.removeChild(ta);
      }
      const toast = btn.parentElement.querySelector('.copy-toast');
      if (toast) {
        toast.style.display = 'inline';
        setTimeout(() => { toast.style.display = 'none'; }, 1200);
      }
      btn.classList.add('ok');
      setTimeout(() => { btn.classList.remove('ok'); }, 1200);
    }
  </script>
</body>
</html>
"""


# -------- Image helpers (doc_id-based) --------
def _get_pil_image(doc_id: int) -> Image.Image:
    """
    Return PIL.Image for given doc_id from HF dataset.
    Supported types:
      - PIL.Image.Image
      - numpy.ndarray
      - base64-encoded str (data URI supported)
    """
    if HF_DATASET is None or not (0 <= int(doc_id) < HF_LENGTH):
        raise FileNotFoundError("Image not available.")
    row = HF_DATASET[int(doc_id)]
    img = row.get(HF_IMAGE_COL)
    if img is None:
        raise FileNotFoundError("Image column is empty.")

    # PIL image
    if isinstance(img, Image.Image):
        return img

    # numpy array
    if np is not None and isinstance(img, np.ndarray):
        return Image.fromarray(img)

    # base64 string (maybe data URI)
    if isinstance(img, str):
        try:
            b64 = img
            if img.startswith("data:"):
                _, _, data = img.partition(",")
                b64 = data
            data_bytes = base64.b64decode(b64)
            return Image.open(io.BytesIO(data_bytes))
        except Exception as e:
            raise ValueError(f"Failed to decode base64 image: {e}")

    raise TypeError(f"Unsupported image type: {type(img)}")


def _encode_image(pil_img: Image.Image, prefer_png: bool = False) -> (bytes, str):
    """
    PIL.Image → (bytes, mimetype)
    Use PNG if alpha channel; otherwise JPEG.
    """
    fmt = "PNG" if prefer_png or pil_img.mode in ("RGBA", "LA") else "JPEG"
    if fmt == "JPEG" and pil_img.mode not in ("RGB", "L"):
        pil_img = pil_img.convert("RGB")
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    buf.seek(0)
    mimetype = "image/png" if fmt == "PNG" else "image/jpeg"
    return buf.read(), mimetype


def _make_thumbnail_bytes(doc_id: int, max_side: int) -> (bytes, str):
    """
    Build thumbnail bytes (with in-memory cache) using doc_id.
    """
    if doc_id in THUMB_CACHE:
        return THUMB_CACHE[doc_id]
    img = _get_pil_image(doc_id)
    thumb = img.copy()
    thumb.thumbnail((max_side, max_side))  # keeps aspect ratio
    data, mimetype = _encode_image(thumb, prefer_png=False)
    THUMB_CACHE[doc_id] = (data, mimetype)
    return data, mimetype


def _send_bytes(data: bytes, mimetype: str, max_age=86400):
    resp = make_response(data)
    resp.headers["Content-Type"] = mimetype
    resp.headers["Cache-Control"] = f"public, max-age={max_age}"
    return resp


def _parse_doc_id_from_data(d) -> int | None:
    """
    Safely parse int(doc_id) from a dict-like item. Returns None if missing/invalid.
    """
    if not isinstance(d, dict):
        return None
    v = d.get("doc_id", None)
    try:
        doc_id = int(v)
        if doc_id < 0:
            return None
        return doc_id
    except Exception:
        return None


# -------- Flask routes --------
@app.route("/")
def index():
    ensure_file(DATA_PATH)
    meta = file_meta(DATA_PATH)
    total_lines = count_lines_cached(DATA_PATH)

    # Filters / paging
    q, sort_key, page_size, compact = common_query_args()
    try:
        page = max(int(request.args.get("page", 1)), 1)
    except Exception:
        page = 1

    # Collect matched records for this page
    start = (page - 1) * page_size
    end = start + page_size

    matched = []
    for _, item in enumerate(iter_jsonl(DATA_PATH), start=1):
        if matches_query(item["data"], item["raw"], q):
            matched.append(item)

    keyfunc = get_sort_keyfunc(sort_key)
    if keyfunc:
        try:
            matched.sort(key=keyfunc)
        except Exception:
            pass

    total_matched = len(matched)
    total_pages = max(1, math.ceil(total_matched / page_size))
    if page > total_pages:
        return redirect(url_for("index", page=total_pages, q=q, sort_key=sort_key, page_size=page_size, compact=int(compact)))

    page_items = matched[start:end]

    # Render model
    items = []
    for i, item in enumerate(page_items, start=start + 1):
        pretty = item["raw"]
        resp_text = None
        resp_marked = None
        doc_id = None
        if item["ok"]:
            try:
                pretty = json.dumps(item["data"], ensure_ascii=False, indent=2)
            except Exception:
                pretty = item["raw"]
            d = item["data"]
            doc_id = _parse_doc_id_from_data(d)
            if isinstance(d, dict):
                fr = d.get("filtered_resps")
                if isinstance(fr, list) and len(fr) > 0 and isinstance(fr[0], str):
                    resp_text = fr[0]
                    resp_marked = highlight_text(resp_text, q)

        has_image = (HF_DATASET is not None) and (doc_id is not None) and (0 <= int(doc_id) < HF_LENGTH)
        chips = build_chips(item["data"]) if item["ok"] else []

        items.append(
            {
                "index": i,  # 1-based for display
                "lineno": item["lineno"],
                "recno": item["recno"],  # keep for reference (not used for image anymore)
                "ok": item["ok"],
                "error": item.get("error"),
                "raw": item["raw"],
                "pretty": pretty,
                "key_preview": Markup(key_preview_html(item["data"])) if item["ok"] else "",
                "resp_text": resp_text,
                "resp_marked": resp_marked,
                "has_image": has_image,
                "chips": chips,
                "doc_id": doc_id,
            }
        )

    def page_link_args(p):
        return {
            "page": p,
            "q": q or None,
            "sort_key": sort_key or None,
            "page_size": page_size if page_size != DEFAULT_PAGE_SIZE else None,
            "compact": int(compact),
        }

    return render_template_string(
        PAGE_TMPL,
        meta=meta,
        total=total_lines,
        size_human=human_size(meta["size"]),
        mtime_human=human_time(meta["mtime"]),
        items=items,
        page=page,
        page_size=page_size,
        max_page_size=MAX_PAGE_SIZE,
        q=q,
        sort_key=sort_key,
        pages=build_pages(page, total_pages),
        page_link_args=page_link_args,
        image_col=HF_IMAGE_COL or "image",
        thumb_max=THUMB_MAX,
        compact=compact,
    )


@app.route("/item/<int:index>")
def item_detail(index):
    ensure_file(DATA_PATH)
    q, sort_key, page_size, compact = common_query_args()

    matched = []
    for _, item in enumerate(iter_jsonl(DATA_PATH), start=1):
        if matches_query(item["data"], item["raw"], q):
            matched.append(item)

    keyfunc = get_sort_keyfunc(sort_key)
    if keyfunc:
        try:
            matched.sort(key=keyfunc)
        except Exception:
            pass

    if index < 1 or index > len(matched):
        abort(404, description="Item not found.")

    chosen = matched[index - 1]
    pretty = chosen["raw"]
    resp_text = None
    resp_marked = None
    doc_id = None
    if chosen["ok"]:
        try:
            pretty = json.dumps(chosen["data"], ensure_ascii=False, indent=2)
        except Exception:
            pretty = chosen["raw"]
        d = chosen["data"]
        doc_id = _parse_doc_id_from_data(d)
        if isinstance(d, dict):
            fr = d.get("filtered_resps")
            if isinstance(fr, list) and len(fr) > 0 and isinstance(fr[0], str):
                resp_text = fr[0]
                resp_marked = highlight_text(resp_text, q)

    has_image = (HF_DATASET is not None) and (doc_id is not None) and (0 <= int(doc_id) < HF_LENGTH)

    meta = file_meta(DATA_PATH)
    return render_template_string(
        DETAIL_TMPL,
        meta=meta,
        index=index,
        lineno=chosen["lineno"],
        ok=chosen["ok"],
        error=chosen.get("error"),
        raw=chosen["raw"],
        pretty=pretty,
        sort_key=sort_key or "",
        resp_text=resp_text,
        resp_marked=resp_marked,
        has_image=has_image,
        doc_id=doc_id,
        image_col=HF_IMAGE_COL or "image",
        thumb_max=THUMB_MAX,
    )


@app.route("/download")
def download():
    ensure_file(DATA_PATH)
    return send_file(DATA_PATH, as_attachment=True, download_name=os.path.basename(DATA_PATH))


# NOTE: routes now take doc_id (NOT recno)
@app.route("/image/<int:doc_id>")
def serve_image(doc_id: int):
    """
    Serve original image for given doc_id from the dataset as PNG/JPEG.
    """
    if HF_DATASET is None or not (0 <= int(doc_id) < HF_LENGTH):
        abort(404, description="Image not available.")
    try:
        pil_img = _get_pil_image(doc_id)
        data, mimetype = _encode_image(pil_img, prefer_png=False)
        return _send_bytes(data, mimetype)
    except Exception as e:
        abort(500, description=f"Error while processing image: {type(e).__name__}: {e}")


@app.route("/thumb/<int:doc_id>")
def serve_thumb(doc_id: int):
    """
    Serve thumbnail image for given doc_id (uses memory cache).
    """
    if HF_DATASET is None or not (0 <= int(doc_id) < HF_LENGTH):
        abort(404, description="Image not available.")
    try:
        data, mimetype = _make_thumbnail_bytes(doc_id, THUMB_MAX)
        return _send_bytes(data, mimetype)
    except Exception as e:
        abort(500, description=f"Error while processing thumbnail: {type(e).__name__}: {e}")


# -------- Entry point / Dataset loader --------
def parse_args():
    ap = argparse.ArgumentParser(description="JSONL + HF image viewer (lightbox/thumbnail/highlight/compact)")
    ap.add_argument("jsonl_path", help="Path to JSONL file")
    ap.add_argument("--host", default="0.0.0.0", help="Host (default: 127.0.0.1)")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "5000")), help="Port (default: 5000)")

    # Hugging Face datasets options
    ap.add_argument("--hf-repo", default=None, help="HuggingFace Datasets repo (e.g., 'beans' or 'user/repo')")
    ap.add_argument("--hf-name", default=None, help="Dataset config name (optional)")
    ap.add_argument("--hf-split", default="train", help="Dataset split (default: train)")
    ap.add_argument("--hf-image-col", default=None, help="Image column name (e.g., 'image', 'img')")

    # Thumbnail options
    ap.add_argument("--thumb-max", type=int, default=THUMB_MAX, help="Max thumbnail side in px (default: 512)")
    return ap.parse_args()


def load_hf_dataset(repo: str, name: str, split: str):
    """
    Load with load_dataset(repo, name=name, split=split). If name is None, default config is used.
    """
    global HF_DATASET, HF_LENGTH
    if not HF_AVAILABLE or load_dataset is None:
        print("[WARN] 'datasets' library not found. Image features disabled.", file=sys.stderr)
        return
    try:
        ds = load_dataset(repo, name=name, split=split)
        HF_DATASET = ds
        HF_LENGTH = len(ds)
    except Exception as e:
        print(f"[WARN] Failed to load dataset: {repo} (name={name}, split={split}) → {type(e).__name__}: {e}", file=sys.stderr)


if __name__ == "__main__":
    args = parse_args()
    DATA_PATH = args.jsonl_path
    if not os.path.isfile(DATA_PATH):
        print(f"[ERROR] File not found: {DATA_PATH}", file=sys.stderr)
        sys.exit(1)

    # Thumbnail size
    THUMB_MAX = max(64, int(args.thumb_max))  # safe minimum

    # HF image settings
    if args.hf_repo and args.hf_image_col:
        HF_IMAGE_COL = args.hf_image_col
        load_hf_dataset(args.hf_repo, args.hf_name, args.hf_split)
        if HF_DATASET is None:
            print("[WARN] Image features are disabled.", file=sys.stderr)
    elif args.hf_repo or args.hf_image_col or args.hf_name:
        print("[WARN] --hf-repo and --hf-image-col must be provided together (add --hf-name if needed). Image features disabled.", file=sys.stderr)

    app.run(host=args.host, port=args.port, debug=False)
