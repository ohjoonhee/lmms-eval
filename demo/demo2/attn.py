import argparse
import base64
import json
import logging
import sys
from html import escape

import numpy as np
from flask import Flask, jsonify, render_template_string, request

# --- Global Data Store ---
# These will be populated at startup by either loading files or generating demo data.
ATTENTIONS = []
TOKENS = []
METADATA = {}

# --- Flask App Initialization ---
app = Flask(__name__)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR) # Suppress boilerplate Flask logging

# --- Data Processing Utilities ---

def extract_head_vectors(step_t, layer_l):
    """
    Extracts attention vectors for a specific token and layer across all heads.
    Handles the two different tensor shapes for initial prompt vs. generated tokens.

    Args:
        step_t (int): The generation step (index in the outer ATTENTIONS list).
        layer_l (int): The layer index.

    Returns:
        np.ndarray: A NumPy array of shape [H, S_t], where H is the number of heads
                    and S_t is the context length at that step.
    """
    try:
        import torch
        attn_tensor = ATTENTIONS[step_t][layer_l].cpu().to(torch.float32)

        # Case 1: Initial prompt processing, shape [1, H, S, S]
        if attn_tensor.ndim == 4 and attn_tensor.shape[2] > 1:
            # We only care about the attention from the last query token to all keys
            head_vectors = attn_tensor[0, :, -1, :].numpy()
        # Case 2: KV-cached generation, shape [1, H, 1, S_t]
        else:
            head_vectors = attn_tensor[0, :, 0, :].numpy()

        return head_vectors
    except ImportError:
        # This path is taken in --demo mode without torch installed
        return ATTENTIONS[step_t][layer_l]


def normalize_per_head(vectors):
    """
    Performs min-max normalization on each head's attention vector.

    Args:
        vectors (np.ndarray): Array of shape [H, S_t].

    Returns:
        np.ndarray: Normalized array of the same shape, with values in [0, 1].
    """
    H, S_t = vectors.shape
    normalized = np.zeros_like(vectors, dtype=np.float32)
    for h in range(H):
        head_vec = vectors[h]
        min_val, max_val = head_vec.min(), head_vec.max()
        if max_val - min_val > 1e-6: # Avoid division by zero for zero-variance heads
            normalized[h] = (head_vec - min_val) / (max_val - min_val)
        else:
            normalized[h] = np.zeros_like(head_vec) # Clamp to zero
    return normalized

def to_base64_float32(arr):
    """
    Encodes a NumPy array of floats into a Base64 string.
    """
    return base64.b64encode(arr.astype(np.float32).tobytes()).decode('utf-8')

def generate_synthetic_data():
    """
    Creates plausible-looking synthetic data for demo mode.
    """
    print("--- Running in Demo Mode ---")
    global ATTENTIONS, TOKENS, METADATA
    L, H, T, S0 = 4, 8, 50, 128
    
    TOKENS = [f"tok_{i}" for i in range(T)]
    
    for t in range(T):
        layer_attns = []
        context_len = S0 + t
        for l in range(L):
            # Generate Dirichlet-like sparse attention
            # The query token attends to a few key tokens strongly
            raw_attn = np.random.dirichlet(np.ones(context_len) * 0.1, size=H)
            layer_attns.append(raw_attn)
        ATTENTIONS.append(layer_attns)

    METADATA = {
        'layers': L,
        'heads': H,
        'steps': T,
        'tokens': TOKENS,
        'last_context': S0 + T -1,
        'prompt_len': S0,
    }
    print(f"Generated demo data: {L} layers, {H} heads, {T} generated tokens, {S0} prompt tokens.")


def load_data_from_files(attn_path, tokens_path):
    """
    Loads attention and token data from .pt files.
    """
    global ATTENTIONS, TOKENS, METADATA
    try:
        import torch
    except ImportError:
        print("Error: 'torch' is required to load data from files.")
        print("Please install it (`pip install torch`) or run in --demo mode.")
        sys.exit(1)

    print(f"Loading attentions from {attn_path}...")
    ATTENTIONS = torch.load(attn_path, map_location=torch.device('cpu'))
    print(f"Loading tokens from {tokens_path}...")
    TOKENS = torch.load(tokens_path, map_location=torch.device('cpu'))
    
    if not isinstance(ATTENTIONS, (list, tuple)) or not all(isinstance(x, (list, tuple)) for x in ATTENTIONS):
        print("Error: attentions.pt should be a List[List[torch.Tensor]]")
        sys.exit(1)

    if not ATTENTIONS:
        print("Error: Attention data is empty.")
        sys.exit(1)

    # Infer metadata
    T = len(ATTENTIONS)
    L = len(ATTENTIONS[0])
    # Shape for t>0 is [1,H,1,S_t], for t=0 is [1,H,S,S]
    first_token_attn = ATTENTIONS[0][0]
    H = first_token_attn.shape[1]
    last_context = ATTENTIONS[-1][0].shape[-1]
    
    # FIX: Correctly and robustly determine the prompt length from the first tensor's shape.
    if first_token_attn.shape[2] > 1: # Case: [1, H, S, S]
        prompt_len = first_token_attn.shape[2]
    else: # Case: [1, H, 1, S]
        prompt_len = first_token_attn.shape[3]

    METADATA = {
        'layers': L,
        'heads': H,
        'steps': T,
        'tokens': TOKENS,
        'last_context': last_context,
        'prompt_len': prompt_len,
    }
    print(f"Loaded data: {L} layers, {H} heads, {T} generated tokens, {prompt_len} prompt tokens.")


# --- Frontend Templates ---

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>KV-Cache Attention Visualizer</title>
    <style>
        :root {
            --bg-color: #111;
            --text-color: #eee;
            --grid-line-color: #444;
            --header-bg: #222;
            --token-bg: #333;
            --token-hover-bg: #555;
            --token-active-bg: #007bff;
            --tooltip-bg: rgba(40, 40, 40, 0.95);
            --modal-bg: rgba(0, 0, 0, 0.7);
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-color);
            margin: 0;
            padding: 0;
            display: flex;
            flex-direction: column;
            height: 100vh;
            overflow: hidden;
        }
        .header {
            background-color: var(--header-bg);
            padding: 10px 20px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.3);
            z-index: 10;
        }
        #token-selector {
            display: flex;
            overflow-x: auto;
            white-space: nowrap;
            padding: 5px 0;
            scrollbar-color: var(--grid-line-color) var(--header-bg);
        }
        .token {
            padding: 5px 10px;
            margin: 0 4px;
            border-radius: 4px;
            background-color: var(--token-bg);
            cursor: pointer;
            transition: background-color 0.2s;
            position: relative;
        }
        .token:hover { background-color: var(--token-hover-bg); }
        .token.active { background-color: var(--token-active-bg); }
        .token .tooltip-text {
            visibility: hidden;
            background-color: var(--tooltip-bg);
            color: #fff;
            text-align: center;
            border-radius: 4px;
            padding: 5px 8px;
            position: absolute;
            z-index: 1;
            bottom: 125%;
            left: 50%;
            transform: translateX(-50%);
            opacity: 0;
            transition: opacity 0.2s;
        }
        .token:hover .tooltip-text { visibility: visible; opacity: 1; }
        .main-content {
            flex-grow: 1;
            display: grid;
            grid-template-columns: 50px 1fr;
            grid-template-rows: 50px 1fr;
            padding: 10px;
            gap: 10px;
        }
        #layer-labels {
            grid-column: 1 / 2;
            grid-row: 2 / 3;
            display: flex;
            flex-direction: column;
            justify-content: space-around;
            align-items: center;
        }
        #head-labels {
            grid-column: 2 / 3;
            grid-row: 1 / 2;
            display: flex;
            justify-content: space-around;
            align-items: center;
        }
        #grid-container {
            grid-column: 2 / 3;
            grid-row: 2 / 3;
            display: grid;
            gap: 5px;
            overflow: auto;
        }
        .grid-cell {
            background-color: #000;
            border: 1px solid var(--grid-line-color);
            border-radius: 4px;
            cursor: pointer;
            transition: border-color 0.2s;
        }
        .grid-cell:hover { border-color: var(--token-active-bg); }
        #modal-overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: var(--modal-bg);
            backdrop-filter: blur(2px);
            display: none;
            justify-content: center;
            align-items: center;
            z-index: 100;
        }
        #modal-content {
            background-color: var(--bg-color);
            padding: 20px;
            border-radius: 8px;
            border: 1px solid var(--grid-line-color);
            box-shadow: 0 5px 15px rgba(0,0,0,0.5);
            display: flex;
            position: relative;
        }
        #modal-canvas { border: 1px solid var(--grid-line-color); }
        .axis-labels {
            position: relative; /* For positioning children */
            width: 150px; /* Give it enough space for text */
            font-size: 10px;
            writing-mode: vertical-rl;
            text-orientation: mixed;
            white-space: nowrap;
            text-align: center;
        }
        /* .key-labels { transform: scale(-1, -1); } REMOVED this problematic rule */
        .query-labels span.active { color: var(--token-active-bg); font-weight: bold; }
        #line-tooltip {
            position: fixed;
            background-color: var(--tooltip-bg);
            padding: 5px 10px;
            border-radius: 4px;
            font-size: 12px;
            pointer-events: none;
            display: none;
            z-index: 101;
        }
        #status {
            padding: 0 20px;
            font-size: 12px;
            color: #888;
        }
    </style>
</head>
<body>
    <div class="header">
        <div id="token-selector"></div>
        <div id="status">Loading metadata...</div>
    </div>
    <div class="main-content">
        <div id="layer-labels"></div>
        <div id="head-labels"></div>
        <div id="grid-container"></div>
    </div>

    <div id="modal-overlay">
        <div id="modal-content">
            <div id="key-axis" class="axis-labels key-labels"></div>
            <canvas id="modal-canvas"></canvas>
            <div id="query-axis" class="axis-labels query-labels"></div>
        </div>
    </div>
    <div id="line-tooltip"></div>

    <script>
        // --- Global State ---
        let META = {};
        let currentTokenIndex = 0;
        let attentionCache = new Map(); // LRU cache for token attention data
        const CACHE_SIZE = 8;

        // --- UI Elements ---
        const tokenSelector = document.getElementById('token-selector');
        const gridContainer = document.getElementById('grid-container');
        const layerLabels = document.getElementById('layer-labels');
        const headLabels = document.getElementById('head-labels');
        const statusEl = document.getElementById('status');
        const modalOverlay = document.getElementById('modal-overlay');
        const modalCanvas = document.getElementById('modal-canvas');
        const lineTooltip = document.getElementById('line-tooltip');

        // --- Initialization ---
        async function init() {
            try {
                const response = await fetch('/meta');
                META = await response.json();
                
                setupUI();
                await selectToken(0);
            } catch (error) {
                console.error("Failed to initialize:", error);
                statusEl.textContent = `Error: Failed to fetch metadata. Is the server running?`;
            }
        }

        function setupUI() {
            // Token selector
            META.tokens.forEach((token, i) => {
                const el = document.createElement('div');
                el.className = 'token';
                el.dataset.index = i;
                el.textContent = escapeHtml(token);
                el.onclick = () => selectToken(i);
                
                const tooltip = document.createElement('span');
                tooltip.className = 'tooltip-text';
                tooltip.textContent = `t=${i}`;
                el.appendChild(tooltip);

                tokenSelector.appendChild(el);
            });

            // Grid and labels
            gridContainer.style.gridTemplateColumns = `repeat(${META.heads}, 1fr)`;
            gridContainer.style.gridTemplateRows = `repeat(${META.layers}, 1fr)`;

            for (let l = 0; l < META.layers; l++) {
                const label = document.createElement('div');
                label.textContent = `L ${l}`;
                layerLabels.appendChild(label);
                for (let h = 0; h < META.heads; h++) {
                    const canvas = document.createElement('canvas');
                    canvas.className = 'grid-cell';
                    canvas.dataset.layer = l;
                    canvas.dataset.head = h;
                    canvas.width = 160;
                    canvas.height = 160;
                    canvas.onclick = () => showEnlargedView(l, h);
                    gridContainer.appendChild(canvas);
                }
            }
            
            for (let h = 0; h < META.heads; h++) {
                const label = document.createElement('div');
                label.textContent = `H ${h}`;
                headLabels.appendChild(label);
            }
        }
        
        // --- Data Fetching & Caching ---
        async function fetchAttentionForToken(t) {
            if (attentionCache.has(t)) {
                const data = attentionCache.get(t);
                attentionCache.delete(t); // Move to end for LRU
                attentionCache.set(t, data);
                return data;
            }

            statusEl.textContent = `Fetching attention for token t=${t}...`;
            const response = await fetch(`/token_attn?t=${t}`);
            const data = await response.json();
            
            // Decode Base64 strings into Float32Arrays
            data.decoded_data = data.data.map(layer =>
                layer.map(b64 => {
                    const byteString = atob(b64);
                    const bytes = new Uint8Array(byteString.length);
                    for (let i = 0; i < byteString.length; i++) {
                        bytes[i] = byteString.charCodeAt(i);
                    }
                    return new Float32Array(bytes.buffer);
                })
            );

            attentionCache.set(t, data);
            if (attentionCache.size > CACHE_SIZE) {
                const oldestKey = attentionCache.keys().next().value;
                attentionCache.delete(oldestKey);
            }
            return data;
        }

        // --- Main Drawing Logic ---
        async function selectToken(t, fromKeyboard = false) {
            if (t < 0 || t >= META.steps) return;
            
            currentTokenIndex = t;

            // Update token selector UI
            document.querySelectorAll('.token').forEach(el => {
                el.classList.toggle('active', parseInt(el.dataset.index) === t);
            });
            
            const activeTokenEl = document.querySelector(`.token[data-index='${t}']`);
            if (activeTokenEl) {
                activeTokenEl.scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'center' });
            }

            try {
                const attnData = await fetchAttentionForToken(t);
                statusEl.textContent = `Visualizing token t=${t} (${escapeHtml(META.tokens[t])}). Context length: ${attnData.context_len}.`;
                requestAnimationFrame(() => drawAllGridCells(attnData));
            } catch (error) {
                console.error(`Failed to fetch or render attention for token ${t}:`, error);
                statusEl.textContent = `Error loading attention for token t=${t}. Check console for details.`;
            }
        }

        function drawAllGridCells(attnData) {
            const canvases = gridContainer.querySelectorAll('canvas');
            canvases.forEach(canvas => {
                const l = parseInt(canvas.dataset.layer);
                const h = parseInt(canvas.dataset.head);
                const headVector = attnData.decoded_data[l][h];
                drawAttentionMap(canvas, headVector, attnData.context_len);
            });
        }
        
        const layerColors = ['#007bff', '#fd7e14', '#28a745', '#dc3545', '#6f42c1', '#20c997'];
        function getLineColor(layer, alpha) {
            const baseColor = layerColors[layer % layerColors.length];
            const r = parseInt(baseColor.slice(1, 3), 16);
            const g = parseInt(baseColor.slice(3, 5), 16);
            const b = parseInt(baseColor.slice(5, 7), 16);
            return `rgba(${r}, ${g}, ${b}, ${alpha})`;
        }

        function drawAttentionMap(canvas, headVector, contextLen) {
            const ctx = canvas.getContext('2d');
            const w = canvas.width;
            const h = canvas.height;
            ctx.clearRect(0, 0, w, h);

            const l = parseInt(canvas.dataset.layer);
            const queryY = h / 2; // Query token is always at the center on the right

            for (let i = 0; i < contextLen; i++) {
                const weight = headVector[i];
                if (weight < 0.01) continue; // Performance: skip drawing faint lines

                const keyY = (i / (contextLen - 1)) * h;
                
                ctx.beginPath();
                ctx.moveTo(0, keyY);
                ctx.lineTo(w, queryY);
                ctx.strokeStyle = getLineColor(l, weight * 0.8);
                ctx.lineWidth = 0.5;
                ctx.stroke();
            }
        }

        // --- Enlarged Tooltip View ---
        let activeEnlarged = { l: -1, h: -1, data: null, rafId: null };

        async function showEnlargedView(l, h) {
            const attnData = await fetchAttentionForToken(currentTokenIndex);
            activeEnlarged = { l, h, data: attnData, rafId: null };
            
            const headVector = attnData.decoded_data[l][h];
            const contextLen = attnData.context_len;

            // Must resize canvas FIRST, so we have the correct height for axis setup
            resizeModalCanvas(contextLen);
            setupModalAxes(contextLen);
            
            modalOverlay.style.display = 'flex';
            drawEnlargedMap(headVector, -1); // Initial draw with no hover
        }

        function setupModalAxes(contextLen) {
            const keyAxis = document.getElementById('key-axis');
            const queryAxis = document.getElementById('query-axis');
            keyAxis.innerHTML = '';
            queryAxis.innerHTML = '';

            const queryTokenAbsoluteIndex = META.prompt_len + currentTokenIndex;
            const h = modalCanvas.height;

            for (let i = 0; i < contextLen; i++) {
                const keyLabel = document.createElement('span');
                const queryLabel = document.createElement('span');
                
                const isPrompt = i < META.prompt_len;
                const genTokenIndex = i - META.prompt_len;

                // Handle potential out-of-bounds access if token list is shorter than expected
                const tokenText = (!isPrompt && genTokenIndex < META.tokens.length) 
                                ? escapeHtml(META.tokens[genTokenIndex])
                                : '';

                const labelText = isPrompt ? `[P ${i}]` : `[G ${genTokenIndex}] ${tokenText}`;
                keyLabel.textContent = labelText;
                queryLabel.textContent = labelText;

                if (i === queryTokenAbsoluteIndex) {
                    queryLabel.classList.add('active');
                }
                
                const yPos = (i / (contextLen - 1)) * h;

                [keyLabel, queryLabel].forEach(label => {
                    label.style.position = 'absolute';
                    label.style.top = `${yPos}px`;
                    label.style.left = '50%';
                    // With writing-mode:vertical-rl, translateY is horizontal, translateX is vertical.
                    // This transform centers the label text on its calculated (x, y) position.
                    label.style.transform = 'translateY(-50%) translateX(-50%)';
                });

                keyAxis.appendChild(keyLabel);
                queryAxis.appendChild(queryLabel);
            }
        }
        
        function resizeModalCanvas(contextLen) {
            const height = Math.min(window.innerHeight * 0.8, contextLen * 15, 2000);
            const width = Math.min(window.innerWidth * 0.6, 800);
            modalCanvas.height = height;
            modalCanvas.width = width;
            document.getElementById('modal-content').style.height = `${height}px`;
        }
        
        function drawEnlargedMap(headVector, hoverIndex) {
            const ctx = modalCanvas.getContext('2d');
            const w = modalCanvas.width;
            const h = modalCanvas.height;
            const contextLen = headVector.length;
            ctx.clearRect(0, 0, w, h);

            const queryTokenAbsoluteIndex = META.prompt_len + currentTokenIndex;
            const queryY = (queryTokenAbsoluteIndex / (contextLen - 1)) * h;

            // Draw all lines
            for (let i = 0; i < contextLen; i++) {
                const weight = headVector[i];
                if (weight < 0.01 && i !== hoverIndex) continue;
                
                const keyY = (i / (contextLen - 1)) * h;

                ctx.beginPath();
                ctx.moveTo(0, keyY);
                ctx.lineTo(w, queryY);
                
                const isHovered = i === hoverIndex;
                ctx.lineWidth = isHovered ? 2.5 : 1;
                ctx.strokeStyle = getLineColor(activeEnlarged.l, isHovered ? 1.0 : weight * 0.8);
                ctx.stroke();
            }
        }

        function handleModalMouseMove(event) {
            if (activeEnlarged.rafId) {
                cancelAnimationFrame(activeEnlarged.rafId);
            }
            activeEnlarged.rafId = requestAnimationFrame(() => {
                const rect = modalCanvas.getBoundingClientRect();
                const y = event.clientY - rect.top;
                
                const contextLen = activeEnlarged.data.context_len;
                const keyIndex = Math.round((y / modalCanvas.height) * (contextLen - 1));

                if (keyIndex >= 0 && keyIndex < contextLen) {
                    const headVector = activeEnlarged.data.decoded_data[activeEnlarged.l][activeEnlarged.h];
                    drawEnlargedMap(headVector, keyIndex);

                    lineTooltip.style.display = 'block';
                    lineTooltip.style.left = `${event.clientX + 15}px`;
                    lineTooltip.style.top = `${event.clientY}px`;
                    const weight = headVector[keyIndex];
                    lineTooltip.innerHTML = `Key Index: ${keyIndex}<br>Weight: ${weight.toFixed(4)}`;
                }
            });
        }
        
        function hideEnlargedView() {
            modalOverlay.style.display = 'none';
            lineTooltip.style.display = 'none';
            if (activeEnlarged.rafId) {
                cancelAnimationFrame(activeEnlarged.rafId);
            }
            activeEnlarged = { l: -1, h: -1, data: null, rafId: null };
        }
        
        // --- Event Listeners ---
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                hideEnlargedView();
            } else if (modalOverlay.style.display !== 'flex') {
                if (e.key === 'ArrowRight') {
                    selectToken(currentTokenIndex + 1, true);
                } else if (e.key === 'ArrowLeft') {
                    selectToken(currentTokenIndex - 1, true);
                }
            }
        });
        
        modalOverlay.addEventListener('click', (e) => {
            if (e.target === modalOverlay) {
                hideEnlargedView();
            }
        });
        
        modalCanvas.addEventListener('mousemove', handleModalMouseMove);
        modalCanvas.addEventListener('mouseleave', () => {
             if (activeEnlarged.rafId) cancelAnimationFrame(activeEnlarged.rafId);
             lineTooltip.style.display = 'none';
             const headVector = activeEnlarged.data.decoded_data[activeEnlarged.l][activeEnlarged.h];
             drawEnlargedMap(headVector, -1); // Redraw without highlight
        });

        // --- Utils ---
        function escapeHtml(unsafe) {
            if (!unsafe) return '';
            return unsafe
                 .replace(/&/g, "&amp;")
                 .replace(/</g, "&lt;")
                 .replace(/>/g, "&gt;")
                 .replace(/"/g, "&quot;")
                 .replace(/'/g, "&#039;");
        }

        // --- Start ---
        document.addEventListener('DOMContentLoaded', init);
    </script>
</body>
</html>
"""

# --- Flask API Endpoints ---

@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template_string(HTML_TEMPLATE)

@app.route('/healthz')
def health_check():
    """Basic health check endpoint."""
    return jsonify({"ok": True})

@app.route('/meta')
def get_meta():
    """Returns metadata about the loaded model and tokens."""
    return jsonify(METADATA)

@app.route('/token_attn')
def get_token_attention():
    """
    Returns the normalized, per-head attention for a given token step,
    with each head's data encoded in Base64.
    """
    try:
        token_t = int(request.args.get('t', 0))
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid token index 't'"}), 400

    if not (0 <= token_t < METADATA['steps']):
        return jsonify({"error": f"Token index {token_t} out of bounds"}), 400
    
    # FIX: Add a try-except block to handle potential data processing errors gracefully.
    try:
        L, H = METADATA['layers'], METADATA['heads']
        
        all_layers_data = []

        # Validate that the data for the requested token step has the correct number of layers.
        if len(ATTENTIONS[token_t]) != L:
            error_msg = f"Data inconsistency at token {token_t}: expected {L} layers, but found {len(ATTENTIONS[token_t])}."
            app.logger.error(error_msg)
            return jsonify({"error": error_msg}), 500

        # The actual context length comes from the tensor shape for the current token 't'.
        raw_vectors = extract_head_vectors(token_t, 0)
        context_len = raw_vectors.shape[1]

        for l in range(L):
            raw_vectors_layer = extract_head_vectors(token_t, l)
            normalized_vectors = normalize_per_head(raw_vectors_layer)
            
            layer_b64 = [to_base64_float32(normalized_vectors[h]) for h in range(H)]
            all_layers_data.append(layer_b64)
        
        response = {
            "t": token_t,
            "context_len": context_len,
            "layers": L,
            "heads": H,
            "data": all_layers_data
        }
        return jsonify(response)
    except Exception as e:
        app.logger.error(f"Error processing token {token_t}: {e}", exc_info=True)
        return jsonify({"error": f"An unexpected error occurred on the server while processing token {token_t}. See server logs for details."}), 500


# --- Main Execution ---

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Visualize multi-layer, multi-head attention from a KV-cached LLM."
    )
    parser.add_argument(
        '--attn_path', type=str, help="Path to the attentions.pt file."
    )
    parser.add_argument(
        '--tokens_path', type=str, help="Path to the tokens.pt file."
    )
    parser.add_argument(
        '--demo', action='store_true', help="Run with synthetic demo data."
    )
    parser.add_argument('--host', type=str, default='127.0.0.1', help="Host to bind to.")
    parser.add_argument('--port', type=int, default=8000, help="Port to listen on.")
    
    args = parser.parse_args()

    if args.demo:
        generate_synthetic_data()
    elif args.attn_path and args.tokens_path:
        load_data_from_files(args.attn_path, args.tokens_path)
    else:
        parser.error("Either --demo or both --attn_path and --tokens_path are required.")

    print(f"\n🚀 Starting server at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)







