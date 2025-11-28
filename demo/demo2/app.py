import argparse
import os
import json
from flask import Flask, render_template_string, jsonify, send_from_directory

# --- Configuration & Setup ---

# Template for the main index page listing available files.
INDEX_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Available Logprob Files</title>
    <style>
        body { font-family: sans-serif; background-color: #f0f2f5; color: #333; padding: 2rem; }
        .container { max-width: 800px; margin: 0 auto; background-color: #ffffff; padding: 2rem; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        h1 { color: #1c1e21; border-bottom: 2px solid #e0e0e0; padding-bottom: 0.5rem; }
        ul { list-style-type: none; padding: 0; }
        li { margin: 0.75rem 0; }
        a { text-decoration: none; color: #007bff; font-size: 1.1rem; transition: color 0.2s; }
        a:hover { color: #0056b3; text-decoration: underline; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Available Logprob Files</h1>
        <p>Select a file to view its visualization:</p>
        <ul>
            {% for filename in filenames %}
            <li><a href="/view/{{ filename }}">{{ filename }}</a></li>
            {% endfor %}
        </ul>
    </div>
</body>
</html>
"""

# The main template which contains the entire client-side application.
CLIENT_SIDE_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Token Logprob Heatmap - {{ filename }}</title>
    <style>
        body { font-family: sans-serif; background-color: #f0f2f5; color: #333; padding: 2rem; margin: 0;}
        .container { max-width: 900px; margin: 0 auto; background-color: #ffffff; padding: 2rem; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); min-height: 80vh; }
        h1 { color: #1c1e21; border-bottom: 2px solid #e0e0e0; padding-bottom: 0.5rem; }
        .token-display { font-family: 'Courier New', Courier, monospace; word-break: break-word; line-height: 1.8; font-size: 16px; }
        .token { position: relative; display: inline-block; padding: 0.1em 0.3em; margin: 0.05em; border-radius: 4px; cursor: default; transition: transform 0.1s ease-in-out; }
        .token .tooltip { visibility: hidden; width: 450px; max-height: 500px; overflow-y: auto; background-color: #2c2c2c; color: #fff; text-align: center; border-radius: 6px; padding: 1rem; position: absolute; z-index: 10; bottom: 125%; left: 50%; margin-left: -225px; opacity: 0; transition: opacity 0.2s; font-size: 12px; line-height: 1.4; }
        .token .tooltip::-webkit-scrollbar { width: 8px; }
        .token .tooltip::-webkit-scrollbar-track { background: #444; border-radius: 4px; }
        .token .tooltip::-webkit-scrollbar-thumb { background: #777; border-radius: 4px; }
        .token .tooltip::-webkit-scrollbar-thumb:hover { background: #999; }
        .token .tooltip svg { font-family: sans-serif; }
        .token .tooltip .bar-label { fill: #ccc; font-size: 11px; dominant-baseline: middle; }
        .token .tooltip .axis-label { fill: #999; font-size: 9px; font-weight: bold; }
        .token .tooltip::after { content: ""; position: absolute; left: 50%; margin-left: -5px; border-width: 5px; border-style: solid; }
        .token .tooltip:not(.flipped)::after { top: 100%; border-color: #2c2c2c transparent transparent transparent; }
        .token .tooltip.flipped { bottom: auto; top: 125%; }
        .token .tooltip.flipped::after { bottom: 100%; border-color: transparent transparent #2c2c2c transparent; }
        .token:hover { transform: scale(1.1); z-index: 5; }
        .token:hover .tooltip { visibility: visible; opacity: 1; }
        .loader { text-align: center; padding: 3rem; font-size: 1.2rem; color: #555; }
    </style>
</head>
<body>
    <div class="container" id="main-container">
        <!-- App content will be rendered here by JavaScript -->
    </div>

    <script>
        document.addEventListener('DOMContentLoaded', () => {
            const FILENAME = {{ filename|tojson }};
            const mainContainer = document.getElementById('main-container');

            // --- Helper Functions (JavaScript implementation) ---
            function getColorForLogprob(logprob, min, max) {
                if (min === max) return 'hsl(120, 70%, 85%)';
                const normalized = (logprob - min) / (max - min) || 0;
                const hue = normalized * 120;
                return `hsl(${hue.toFixed(2)}, 75%, 85%)`;
            }

            function escapeHtml(str) {
                return str.replace(/[&<>"']/g, match => ({
                    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
                }[match]));
            }

            function formatTokenForDisplay(tokenStr) {
                const numBreaks = (tokenStr.match(/\\n/g) || []).length;
                let safeToken = escapeHtml(tokenStr);
                safeToken = safeToken.replace(/ /g, '·').replace(/\\n/g, '↵');
                if (!tokenStr.trim() && !safeToken) {
                    safeToken = '·'.repeat(tokenStr.length);
                }
                return { visible: safeToken, numBreaks };
            }

            function formatTokenForGraph(tokenStr) {
                let safeToken = escapeHtml(tokenStr);
                safeToken = safeToken.replace(/ /g, '·').replace(/\\n/g, '↵');
                if (!tokenStr.trim()) return '·'.repeat(tokenStr.length);
                return safeToken;
            }

            function prepareGraphData(logprobDict, sampledTokenId, minLpOverall, maxLpOverall) {
                let alternatives = Object.entries(logprobDict).map(([id, details]) => ({
                    tokenId: parseInt(id, 10), logprob: details.logprob,
                    decodedToken: details.decoded_token, rank: details.rank || 999
                }));
                alternatives.sort((a, b) => a.rank - b.rank);
                if (alternatives.length === 0) return { graphBars: [], graphMeta: {} };

                const SVG_WIDTH = 450;
                const HEADER_HEIGHT = 30, FOOTER_HEIGHT = 20;
                const BAR_HEIGHT = 14, BAR_PADDING = 6;
                const LABEL_AREA_WIDTH = 80, VALUE_AREA_WIDTH = 50;
                const GRAPH_AREA_X_START = LABEL_AREA_WIDTH + 5;
                const GRAPH_AREA_WIDTH = SVG_WIDTH - GRAPH_AREA_X_START - VALUE_AREA_WIDTH - 20;

                const lpRange = maxLpOverall - minLpOverall || 1.0;
                const totalBarHeight = alternatives.length * (BAR_HEIGHT + BAR_PADDING);
                const svgHeight = HEADER_HEIGHT + totalBarHeight + FOOTER_HEIGHT;

                const graphBars = alternatives.map((alt, i) => {
                    const normalizedLp = (alt.logprob - minLpOverall) / lpRange;
                    const barWidth = Math.max(1, normalizedLp * GRAPH_AREA_WIDTH);
                    return `
                        <text x="${GRAPH_AREA_X_START - 5}" y="${HEADER_HEIGHT + i * (BAR_HEIGHT + BAR_PADDING) + BAR_HEIGHT / 2}" text-anchor="end" class="bar-label">${formatTokenForGraph(alt.decodedToken)}</text>
                        <rect x="${GRAPH_AREA_X_START}" y="${HEADER_HEIGHT + i * (BAR_HEIGHT + BAR_PADDING)}" width="${barWidth}" height="${BAR_HEIGHT}" fill="${alt.tokenId === sampledTokenId ? '#4CAF50' : '#555'}" />
                        <text x="${GRAPH_AREA_X_START + barWidth + 5}" y="${HEADER_HEIGHT + i * (BAR_HEIGHT + BAR_PADDING) + BAR_HEIGHT / 2}" text-anchor="start" class="bar-label">${alt.logprob.toFixed(4)}</text>
                    `;
                });

                const axisY = HEADER_HEIGHT + totalBarHeight;
                const graphMeta = {
                    minLp: minLpOverall, maxLp: maxLpOverall,
                    svgHeight: svgHeight, axisY: axisY,
                    axisXStart: GRAPH_AREA_X_START,
                    axisXEnd: GRAPH_AREA_X_START + GRAPH_AREA_WIDTH
                };
                return { graphBars, graphMeta };
            }
            
            // --- Application Logic ---
            function showLoading() {
                mainContainer.innerHTML = `<h1>Loading ${escapeHtml(FILENAME)}...</h1><div class="loader">Fetching and processing file. This may take a moment for large files.</div>`;
            }

            function renderVisualization(fileContent) {
                // CORRECTED: Split by a regex that handles both \n and \r\n line endings.
                const lines = fileContent.trim().split(/\\r?\\n/);
                
                const data = [];
                for (const line of lines) {
                    if (line.trim()) {
                        try {
                            data.push(JSON.parse(line));
                        } catch (e) {
                            console.error("Failed to parse JSON line:", line, e);
                        }
                    }
                }

                if (data.length === 0) {
                     mainContainer.innerHTML = `<h1>Error</h1><p>Could not parse any valid data from the file: ${escapeHtml(FILENAME)}</p>`;
                     return;
                }
                
                let minLpOverall = Infinity, maxLpOverall = -Infinity;
                let minLogprobMain = Infinity, maxLogprobMain = -Infinity;

                data.forEach(d => {
                    const mainLp = d.logprob[d.token_id].logprob;
                    minLogprobMain = Math.min(minLogprobMain, mainLp);
                    maxLogprobMain = Math.max(maxLogprobMain, mainLp);
                    Object.values(d.logprob).forEach(details => {
                        minLpOverall = Math.min(minLpOverall, details.logprob);
                        maxLpOverall = Math.max(maxLpOverall, details.logprob);
                    });
                });

                const content = data.map(d => {
                    const mainLp = d.logprob[d.token_id].logprob;
                    const { visible, numBreaks } = formatTokenForDisplay(d.token);
                    const color = getColorForLogprob(mainLp, minLogprobMain, maxLogprobMain);
                    const { graphBars, graphMeta } = prepareGraphData(d.logprob, d.token_id, minLpOverall, maxLpOverall);
                    const breaks = '<br>'.repeat(numBreaks);

                    return `
                        <span class="token" style="background-color: ${color};">
                            ${visible}
                            <span class="tooltip">
                                <svg width="100%" height="${graphMeta.svgHeight}" viewbox="0 0 450 ${graphMeta.svgHeight}">
                                    <text x="225" y="15" text-anchor="middle" fill="white" font-size="12">Next Token Logprob Distribution (Ranked)</text>
                                    ${graphBars.join('')}
                                    <line x1="${graphMeta.axisXStart}" y1="${graphMeta.axisY}" x2="${graphMeta.axisXEnd}" y2="${graphMeta.axisY}" stroke="#555" stroke-width="1"/>
                                    <text x="${graphMeta.axisXStart}" y="${graphMeta.axisY + 10}" text-anchor="start" class="axis-label">${graphMeta.minLp.toFixed(1)}</text>
                                    <text x="${graphMeta.axisXEnd}" y="${graphMeta.axisY + 10}" text-anchor="end" class="axis-label">${graphMeta.maxLp.toFixed(1)}</text>
                                </svg>
                            </span>
                        </span>${breaks}`;
                }).join('');
                
                mainContainer.innerHTML = `
                    <h1>Token Logprob Heatmap: ${escapeHtml(FILENAME)}</h1>
                    <p>Hover over any token to see the log probability distribution of potential next tokens.</p>
                    <div class="token-display">${content}</div>`;

                mainContainer.querySelectorAll('.token').forEach(token => {
                    const tooltip = token.querySelector('.tooltip');
                    if (!tooltip) return;
                    token.addEventListener('mouseenter', () => {
                        const rect = tooltip.getBoundingClientRect();
                        if (rect.top < 0) tooltip.classList.add('flipped');
                    });
                    token.addEventListener('mouseleave', () => {
                        tooltip.classList.remove('flipped');
                    });
                });
            }

            async function loadAndRender() {
                showLoading();
                try {
                    const response = await fetch(`/api/data/${FILENAME}`);
                    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
                    const fileContent = await response.text();
                    renderVisualization(fileContent);
                } catch (error) {
                    mainContainer.innerHTML = `<h1>Error</h1><p>Could not load or render file: ${escapeHtml(FILENAME)}</p><p>${error.toString()}</p>`;
                }
            }
            
            loadAndRender();
        });
    </script>
</body>
</html>
"""

# Global dictionary to map filenames to their full paths
FILE_PATHS = {}
app = Flask(__name__)

# --- Flask Routes ---


@app.route("/")
def home():
    """Serves the main index page listing available files."""
    filenames = sorted(list(FILE_PATHS.keys()))
    return render_template_string(INDEX_TEMPLATE, filenames=filenames)


@app.route("/view/<path:filename>")
def view_file(filename):
    """Serves the client-side application shell for a specific file."""
    if filename not in FILE_PATHS:
        return "File not found", 404
    return render_template_string(CLIENT_SIDE_TEMPLATE, filename=filename)


@app.route("/api/data/<path:filename>")
def get_file_data(filename):
    """API endpoint to serve the raw content of a specific .jsonl file."""
    filepath = FILE_PATHS.get(filename)
    if not filepath or not os.path.exists(filepath):
        return "File not found", 404

    directory = os.path.dirname(filepath)
    base_filename = os.path.basename(filepath)
    return send_from_directory(directory, base_filename)


# --- Main Execution Block ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch a Flask web server to visualize token logprobs. All rendering is done client-side.")
    parser.add_argument("input_path", type=str, help="Path to a single .jsonl file or a directory containing .jsonl files.")
    args = parser.parse_args()

    input_path = args.input_path
    files_to_scan = []

    if not os.path.exists(input_path):
        print(f"Error: Path does not exist: {input_path}")
        exit(1)

    if os.path.isdir(input_path):
        for filename in os.listdir(input_path):
            if filename.endswith(".jsonl"):
                files_to_scan.append(os.path.join(input_path, filename))
    elif os.path.isfile(input_path) and input_path.endswith(".jsonl"):
        files_to_scan.append(input_path)

    if not files_to_scan:
        print(f"Error: No .jsonl files found at the specified path: {input_path}")
        exit(1)

    # At launch, just find the files and store their paths.
    for filepath in files_to_scan:
        filename = os.path.basename(filepath)
        FILE_PATHS[filename] = filepath

    print("-" * 50)
    print(f"Found {len(FILE_PATHS)} file(s). Starting server.")
    print("All processing and rendering will be done in the browser.")
    print("-" * 50)
    print("Open http://127.0.0.1:5000 in your web browser.")
    print("Press CTRL+C to stop the server.")
    print("-" * 50)

    app.run(host="0.0.0.0", port=5000)
