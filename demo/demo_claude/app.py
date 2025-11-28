#!/usr/bin/env python3
"""
Single-file Flask app for visualizing multi-layer, multi-head attention
from KV-cached LLM generation.

Usage:
    python app.py --attn_path /path/to/attentions.pt \\
                  --tokens_path /path/to/tokens.pt \\
                  [--host 127.0.0.1] [--port 8000] [--demo]
"""

import argparse
import base64
from typing import Any

import numpy as np
import torch
from flask import Flask, jsonify, render_template_string, request

# Global state
ATTENTIONS = None  # List[List[torch.Tensor]]
TOKENS = None  # List[str]
LAYERS = 0
HEADS = 0
STEPS = 0
LAST_CONTEXT = 0


def extract_head_vectors(step_t: int, layer_l: int) -> np.ndarray:
    """
    Extract attention vectors for all heads at given step and layer.

    Returns:
        np.ndarray of shape [H, S_t] where H = heads, S_t = context length
    """
    attn_tensor = ATTENTIONS[step_t][layer_l]  # [1, H, Q, K]

    # Handle different shapes based on KV-cache
    if attn_tensor.shape[2] > 1:  # [1, H, S, S] - first step
        # Extract last query row
        vectors = attn_tensor[0, :, -1, :].cpu().float().numpy()
    else:  # [1, H, 1, S_t] - subsequent steps
        vectors = attn_tensor[0, :, 0, :].cpu().float().numpy()

    return vectors  # [H, S_t]


def normalize_per_head(x: np.ndarray) -> np.ndarray:
    """
    Normalize each head's attention weights to [0, 1] using min-max scaling.

    Args:
        x: Array of shape [H, S_t]

    Returns:
        Normalized array of shape [H, S_t]
    """
    result = np.zeros_like(x)
    for h in range(x.shape[0]):
        head_vals = x[h]
        min_val = head_vals.min()
        max_val = head_vals.max()

        if max_val - min_val > 1e-8:  # Avoid division by zero
            result[h] = (head_vals - min_val) / (max_val - min_val)
        else:
            result[h] = 0.5  # Uniform if no variance

    return result


def to_base64_float32(arr: np.ndarray) -> str:
    """Convert numpy array to base64-encoded Float32Array."""
    return base64.b64encode(arr.astype(np.float32).tobytes()).decode("utf-8")


def generate_demo_data() -> None:
    """Generate synthetic attention data for demo mode."""
    global ATTENTIONS, TOKENS, LAYERS, HEADS, STEPS, LAST_CONTEXT

    LAYERS = 4
    HEADS = 8
    STEPS = 5
    S0 = 128

    TOKENS = [f"tok{i}" for i in range(STEPS)]
    ATTENTIONS = []

    for t in range(STEPS):
        step_attns = []
        current_context = S0 + t

        for _ in range(LAYERS):
            if t == 0:
                # First step: [1, H, S0, S0]
                shape = (1, HEADS, S0, S0)
            else:
                # Subsequent steps: [1, H, 1, current_context]
                shape = (1, HEADS, 1, current_context)

            # Generate Dirichlet-like attention (random but sums to 1)
            attn = torch.rand(shape)
            attn = attn / attn.sum(dim=-1, keepdim=True)

            step_attns.append(attn)

        ATTENTIONS.append(step_attns)

    LAST_CONTEXT = S0 + STEPS - 1


def load_data(attn_path: str, tokens_path: str) -> None:
    """Load attention tensors and tokens from disk."""
    global ATTENTIONS, TOKENS, LAYERS, HEADS, STEPS, LAST_CONTEXT

    ATTENTIONS = torch.load(attn_path, map_location="cpu")
    TOKENS = torch.load(tokens_path, map_location="cpu")

    STEPS = len(TOKENS)
    LAYERS = len(ATTENTIONS[0])
    HEADS = ATTENTIONS[0][0].shape[1]

    # Get last context length from final step
    LAST_CONTEXT = ATTENTIONS[-1][0].shape[-1]

    print(f"Loaded data: {STEPS} steps, {LAYERS} layers, {HEADS} heads")
    print(f"First step shape: {ATTENTIONS[0][0].shape}")
    print(f"Last step shape: {ATTENTIONS[-1][0].shape}")
    print(f"Last context length: {LAST_CONTEXT}")


# Flask app
app = Flask(__name__)


@app.route("/")
def index() -> str:
    """Serve the main HTML page with inline CSS and JavaScript."""
    return render_template_string(HTML_TEMPLATE)


@app.route("/meta")
def meta() -> Any:
    """Return metadata about the loaded attention data."""
    return jsonify(
        {
            "layers": LAYERS,
            "heads": HEADS,
            "steps": STEPS,
            "last_context": LAST_CONTEXT,
            "tokens": TOKENS,
        }
    )


@app.route("/token_attn")
def token_attn() -> Any:
    """
    Return normalized attention data for a specific token.

    Query params:
        t: Token index (0 to STEPS-1)

    Returns:
        JSON with base64-encoded Float32 arrays for each layer/head
    """
    t = int(request.args.get("t", 0))

    if t < 0 or t >= STEPS:
        return jsonify({"error": "Invalid token index"}), 400

    # Get context length for this step
    context_len = ATTENTIONS[t][0].shape[-1]

    # Extract and normalize attention for all layers
    data = []
    for layer in range(LAYERS):
        head_vectors = extract_head_vectors(t, layer)  # [H, S_t]
        normalized = normalize_per_head(head_vectors)  # [H, S_t]

        # Convert each head to base64
        layer_data = [to_base64_float32(normalized[h]) for h in range(HEADS)]
        data.append(layer_data)

    return jsonify(
        {
            "t": t,
            "context_len": context_len,
            "layers": LAYERS,
            "heads": HEADS,
            "data": data,
        }
    )


@app.route("/healthz")
def healthz() -> Any:
    """Health check endpoint."""
    return jsonify({"ok": True})


# HTML Template with inline CSS and JavaScript
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Attention Visualizer</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            background: #000;
            color: #fff;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            overflow-x: hidden;
        }

        #container {
            display: flex;
            flex-direction: column;
            height: 100vh;
        }

        #header {
            padding: 15px 20px;
            background: #111;
            border-bottom: 1px solid #333;
        }

        #token-selector {
            display: flex;
            align-items: center;
            gap: 10px;
            overflow-x: auto;
            padding: 10px 0;
        }

        .token-btn {
            padding: 8px 16px;
            background: #222;
            border: 2px solid #444;
            color: #aaa;
            cursor: pointer;
            white-space: nowrap;
            border-radius: 4px;
            transition: all 0.2s;
            font-size: 13px;
        }

        .token-btn:hover {
            background: #333;
            border-color: #666;
        }

        .token-btn.active {
            background: #0066cc;
            border-color: #0088ff;
            color: #fff;
        }

        #grid-container {
            flex: 1;
            overflow: auto;
            padding: 20px;
        }

        #grid-wrapper {
            display: inline-block;
            min-width: 100%;
        }

        #grid {
            display: grid;
            gap: 4px;
            grid-auto-flow: row;
        }

        .grid-cell {
            position: relative;
            cursor: pointer;
            border: 1px solid #333;
            border-radius: 2px;
            overflow: hidden;
            transition: transform 0.1s;
        }

        .grid-cell:hover {
            transform: scale(1.02);
            border-color: #666;
            z-index: 10;
        }

        .grid-cell canvas {
            display: block;
            width: 100%;
            height: 100%;
        }

        #labels {
            position: relative;
        }

        .layer-label {
            position: absolute;
            left: -40px;
            font-size: 11px;
            color: #888;
            text-align: right;
            width: 35px;
        }

        .head-label {
            text-align: center;
            font-size: 11px;
            color: #888;
            margin-bottom: 5px;
        }

        #modal {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.85);
            backdrop-filter: blur(2px);
            z-index: 1000;
            align-items: center;
            justify-content: center;
        }

        #modal.active {
            display: flex;
        }

        #modal-content {
            background: #111;
            border: 2px solid #444;
            border-radius: 8px;
            padding: 20px;
            max-width: 90vw;
            max-height: 90vh;
            overflow: auto;
            position: relative;
        }

        #modal-header {
            margin-bottom: 15px;
            font-size: 16px;
            color: #0088ff;
        }

        #modal-canvas-wrapper {
            position: relative;
            display: flex;
            gap: 10px;
        }

        #modal-left-labels,
        #modal-right-labels {
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            font-size: 10px;
            color: #aaa;
            max-height: 600px;
            overflow-y: auto;
        }

        #modal-canvas {
            border: 1px solid #333;
            cursor: crosshair;
        }

        #modal-tooltip {
            position: fixed;
            background: rgba(0, 0, 0, 0.9);
            border: 1px solid #666;
            padding: 6px 10px;
            border-radius: 4px;
            font-size: 12px;
            pointer-events: none;
            display: none;
            z-index: 1001;
        }

        #loading {
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            font-size: 18px;
            color: #0088ff;
        }

        .spinner {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid #333;
            border-top-color: #0088ff;
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
            margin-right: 10px;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div id="container">
        <div id="header">
            <h1 style="margin-bottom: 10px;">Attention Visualizer</h1>
            <div id="token-selector"></div>
        </div>
        <div id="grid-container">
            <div id="grid-wrapper">
                <div id="grid"></div>
            </div>
        </div>
    </div>

    <div id="modal">
        <div id="modal-content">
            <div id="modal-header"></div>
            <div id="modal-canvas-wrapper">
                <div id="modal-left-labels"></div>
                <canvas id="modal-canvas"></canvas>
                <div id="modal-right-labels"></div>
            </div>
        </div>
    </div>

    <div id="modal-tooltip"></div>
    <div id="loading">
        <span class="spinner"></span>
        Loading...
    </div>

    <script>
        // Global state
        let META = null;
        let CURRENT_TOKEN = 0;
        let CACHE = new Map(); // LRU cache for token attention data
        const CACHE_SIZE = 8;
        const CELL_SIZE = 160;
        const MODAL_CANVAS_WIDTH = 600;
        const MODAL_CANVAS_HEIGHT = 600;

        // LRU Cache implementation
        function cacheGet(key) {
            if (!CACHE.has(key)) return null;
            const value = CACHE.get(key);
            CACHE.delete(key);
            CACHE.set(key, value);
            return value;
        }

        function cacheSet(key, value) {
            if (CACHE.has(key)) {
                CACHE.delete(key);
            } else if (CACHE.size >= CACHE_SIZE) {
                const firstKey = CACHE.keys().next().value;
                CACHE.delete(firstKey);
            }
            CACHE.set(key, value);
        }

        // Decode base64 Float32Array
        function decodeFloat32(base64Str) {
            const binaryString = atob(base64Str);
            const bytes = new Uint8Array(binaryString.length);
            for (let i = 0; i < binaryString.length; i++) {
                bytes[i] = binaryString.charCodeAt(i);
            }
            return new Float32Array(bytes.buffer);
        }

        // Initialize app
        async function init() {
            const response = await fetch('/meta');
            META = await response.json();

            setupTokenSelector();
            setupGrid();
            await loadToken(0);

            document.getElementById('loading').style.display = 'none';

            // Keyboard navigation
            document.addEventListener('keydown', (e) => {
                if (e.key === 'ArrowLeft' && CURRENT_TOKEN > 0) {
                    loadToken(CURRENT_TOKEN - 1);
                } else if (e.key === 'ArrowRight' && CURRENT_TOKEN < META.steps - 1) {
                    loadToken(CURRENT_TOKEN + 1);
                } else if (e.key === 'Escape') {
                    closeModal();
                }
            });

            // Close modal on outside click
            document.getElementById('modal').addEventListener('click', (e) => {
                if (e.target.id === 'modal') {
                    closeModal();
                }
            });
        }

        // Setup token selector
        function setupTokenSelector() {
            const selector = document.getElementById('token-selector');
            META.tokens.forEach((token, idx) => {
                const btn = document.createElement('button');
                btn.className = 'token-btn';
                btn.textContent = token;
                btn.title = `Token ${idx}: ${token}`;
                btn.onclick = () => loadToken(idx);
                selector.appendChild(btn);
            });
        }

        // Setup grid layout
        function setupGrid() {
            const grid = document.getElementById('grid');
            grid.style.gridTemplateColumns = `repeat(${META.heads}, ${CELL_SIZE}px)`;
            grid.style.gridTemplateRows = `repeat(${META.layers}, ${CELL_SIZE}px)`;

            // Create grid cells
            for (let layer = 0; layer < META.layers; layer++) {
                for (let head = 0; head < META.heads; head++) {
                    const cell = document.createElement('div');
                    cell.className = 'grid-cell';
                    cell.dataset.layer = layer;
                    cell.dataset.head = head;

                    const canvas = document.createElement('canvas');
                    canvas.width = CELL_SIZE;
                    canvas.height = CELL_SIZE;
                    cell.appendChild(canvas);

                    cell.onclick = () => openModal(layer, head);

                    grid.appendChild(cell);
                }
            }

            // Add labels
            const headLabels = document.createElement('div');
            headLabels.style.display = 'grid';
            headLabels.style.gridTemplateColumns = `repeat(${META.heads}, ${CELL_SIZE}px)`;
            headLabels.style.gap = '4px';
            headLabels.style.marginBottom = '5px';

            for (let h = 0; h < META.heads; h++) {
                const label = document.createElement('div');
                label.className = 'head-label';
                label.textContent = `Head ${h}`;
                headLabels.appendChild(label);
            }

            document.getElementById('grid-wrapper').insertBefore(
                headLabels,
                document.getElementById('grid')
            );
        }

        // Load token attention data
        async function loadToken(tokenIdx) {
            CURRENT_TOKEN = tokenIdx;

            // Update token selector
            document.querySelectorAll('.token-btn').forEach((btn, idx) => {
                btn.classList.toggle('active', idx === tokenIdx);
            });

            // Check cache
            let data = cacheGet(tokenIdx);
            if (!data) {
                const response = await fetch(`/token_attn?t=${tokenIdx}`);
                data = await response.json();
                cacheSet(tokenIdx, data);
            }

            // Decode and render all cells
            for (let layer = 0; layer < META.layers; layer++) {
                for (let head = 0; head < META.heads; head++) {
                    const cell = document.querySelector(
                        `.grid-cell[data-layer="${layer}"][data-head="${head}"]`
                    );
                    const canvas = cell.querySelector('canvas');
                    const weights = decodeFloat32(data.data[layer][head]);

                    renderAttentionCell(canvas, weights, data.context_len);
                }
            }
        }

        // Render attention in grid cell
        function renderAttentionCell(canvas, weights, contextLen) {
            const ctx = canvas.getContext('2d');
            const width = canvas.width;
            const height = canvas.height;

            ctx.clearRect(0, 0, width, height);

            const leftX = width * 0.1;
            const rightX = width * 0.9;
            const topY = 0;
            const bottomY = height;

            // Sample points for performance (show at most 500 lines)
            const step = Math.max(1, Math.floor(contextLen / 500));

            for (let i = 0; i < contextLen; i += step) {
                const weight = weights[i];
                const yPos = topY + (i / contextLen) * (bottomY - topY);

                ctx.strokeStyle = `rgba(100, 150, 255, ${weight * 0.7})`;
                ctx.lineWidth = weight > 0.5 ? 1.5 : 0.8;

                ctx.beginPath();
                ctx.moveTo(leftX, yPos);
                ctx.lineTo(rightX, bottomY - 10);
                ctx.stroke();
            }
        }

        // Open enlarged modal view
        async function openModal(layer, head) {
            const data = cacheGet(CURRENT_TOKEN);
            if (!data) return;

            const weights = decodeFloat32(data.data[layer][head]);
            const contextLen = data.context_len;

            // Setup modal header
            document.getElementById('modal-header').textContent =
                `Layer ${layer} | Head ${head} | Token ${CURRENT_TOKEN}`;

            // Setup canvas
            const canvas = document.getElementById('modal-canvas');
            canvas.width = MODAL_CANVAS_WIDTH;
            canvas.height = MODAL_CANVAS_HEIGHT;

            // Store data for hover interactions
            canvas.dataset.layer = layer;
            canvas.dataset.head = head;
            canvas.dataset.contextLen = contextLen;

            renderModalAttention(canvas, weights, contextLen);

            // Setup token labels (placeholder - tokens would need full sequence)
            setupModalLabels(contextLen);

            // Show modal
            document.getElementById('modal').classList.add('active');

            // Add hover handler
            canvas.onmousemove = (e) => handleModalHover(e, canvas, weights, contextLen);
            canvas.onmouseleave = () => {
                document.getElementById('modal-tooltip').style.display = 'none';
            };
        }

        // Render attention in modal
        function renderModalAttention(canvas, weights, contextLen) {
            const ctx = canvas.getContext('2d');
            const width = canvas.width;
            const height = canvas.height;

            ctx.clearRect(0, 0, width, height);

            const leftX = 50;
            const rightX = width - 50;
            const topY = 20;
            const bottomY = height - 20;
            const queryY = bottomY;

            // Draw all lines
            for (let i = 0; i < contextLen; i++) {
                const weight = weights[i];
                const keyY = topY + (i / contextLen) * (bottomY - topY);

                ctx.strokeStyle = `rgba(100, 150, 255, ${weight * 0.5})`;
                ctx.lineWidth = weight > 0.7 ? 2 : (weight > 0.3 ? 1.2 : 0.6);

                ctx.beginPath();
                ctx.moveTo(leftX, keyY);
                ctx.lineTo(rightX, queryY);
                ctx.stroke();
            }
        }

        // Handle modal hover
        function handleModalHover(e, canvas, weights, contextLen) {
            const rect = canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;

            const topY = 20;
            const bottomY = canvas.height - 20;

            // Find closest key token
            const relativeY = (y - topY) / (bottomY - topY);
            const keyIdx = Math.floor(relativeY * contextLen);

            if (keyIdx >= 0 && keyIdx < contextLen) {
                const tooltip = document.getElementById('modal-tooltip');
                tooltip.style.display = 'block';
                tooltip.style.left = `${e.clientX + 15}px`;
                tooltip.style.top = `${e.clientY + 15}px`;
                tooltip.textContent = `Key: ${keyIdx} | Weight: ${weights[keyIdx].toFixed(4)}`;

                // Highlight line
                renderModalAttention(canvas, weights, contextLen);
                highlightLine(canvas, keyIdx, weights[keyIdx], contextLen);
            }
        }

        // Highlight specific attention line
        function highlightLine(canvas, keyIdx, weight, contextLen) {
            const ctx = canvas.getContext('2d');
            const width = canvas.width;
            const height = canvas.height;

            const leftX = 50;
            const rightX = width - 50;
            const topY = 20;
            const bottomY = height - 20;
            const queryY = bottomY;

            const keyY = topY + (keyIdx / contextLen) * (bottomY - topY);

            ctx.strokeStyle = `rgba(255, 200, 50, ${Math.min(1, weight * 1.5)})`;
            ctx.lineWidth = 3;

            ctx.beginPath();
            ctx.moveTo(leftX, keyY);
            ctx.lineTo(rightX, queryY);
            ctx.stroke();
        }

        // Setup modal token labels
        function setupModalLabels(contextLen) {
            const leftLabels = document.getElementById('modal-left-labels');
            const rightLabels = document.getElementById('modal-right-labels');

            leftLabels.innerHTML = '';
            rightLabels.innerHTML = '';

            // Show sampled labels for performance
            const labelCount = Math.min(20, contextLen);
            const step = Math.floor(contextLen / labelCount);

            for (let i = 0; i < contextLen; i += step) {
                const leftLabel = document.createElement('div');
                leftLabel.textContent = `K${i}`;
                leftLabel.style.fontSize = '9px';
                leftLabels.appendChild(leftLabel);

                const rightLabel = document.createElement('div');
                rightLabel.textContent = `Q${i}`;
                rightLabel.style.fontSize = '9px';
                rightLabels.appendChild(rightLabel);
            }
        }

        // Close modal
        function closeModal() {
            document.getElementById('modal').classList.remove('active');
            document.getElementById('modal-tooltip').style.display = 'none';
        }

        // Start app
        init();
    </script>
</body>
</html>
"""


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Visualize multi-layer, multi-head attention"
    )
    parser.add_argument("--attn_path", help="Path to attentions.pt")
    parser.add_argument("--tokens_path", help="Path to tokens.pt")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--demo", action="store_true", help="Run in demo mode")

    args = parser.parse_args()

    if args.demo:
        print("Running in demo mode with synthetic data...")
        generate_demo_data()
    else:
        if not args.attn_path or not args.tokens_path:
            parser.error("--attn_path and --tokens_path are required")

        print(f"Loading data from {args.attn_path} and {args.tokens_path}...")
        load_data(args.attn_path, args.tokens_path)

    print(f"\nStarting server on http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop\n")

    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()