import math
import io
import base64
from typing import List, Optional
from mcp.server.fastmcp import FastMCP
from mcp.types import ImageContent
from PIL import Image
import os
import requests

app = FastMCP("qwen-image-tools")

# --- Helper Functions (Ported from qwen3vl_image_tool.py) ---


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def smart_resize(height: int, width: int, factor: int = 32, min_pixels: int = 56 * 56, max_pixels: int = 12845056) -> tuple[int, int]:
    """Smart resize image dimensions based on factor and pixel constraints"""
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def maybe_resize_bbox(left, top, right, bottom, img_width, img_height):
    """Resize bbox to ensure it's valid"""
    left = max(0, left)
    top = max(0, top)
    right = min(img_width, right)
    bottom = min(img_height, bottom)

    height = bottom - top
    width = right - left
    if height < 32 or width < 32:
        center_x = (left + right) / 2.0
        center_y = (top + bottom) / 2.0
        ratio = 32 / min(height, width)
        new_half_height = math.ceil(height * ratio * 0.5)
        new_half_width = math.ceil(width * ratio * 0.5)
        new_left = math.floor(center_x - new_half_width)
        new_right = math.ceil(center_x + new_half_width)
        new_top = math.floor(center_y - new_half_height)
        new_bottom = math.ceil(center_y + new_half_height)

        # Ensure the resized bbox is within image bounds
        new_left = max(0, new_left)
        new_top = max(0, new_top)
        new_right = min(img_width, new_right)
        new_bottom = min(img_height, new_bottom)

        new_height = new_bottom - new_top
        new_width = new_right - new_left

        if new_height > 32 and new_width > 32:
            return [new_left, new_top, new_right, new_bottom]
    return [left, top, right, bottom]


# --- MCP Tool Implementation ---


@app.tool(name="image_zoom_in", description="Zoom in on a specific region of an image by cropping it based on a bounding box (bbox) and an optional object label")
def image_zoom_in(image_path: str, bbox_2d: List[float], label: Optional[str] = None):
    """
    Zoom in on a specific region of an image.

    Args:
        image_path: Path to the image file (local path).
        bbox_2d: The bounding box of the region to zoom in, as [x1, y1, x2, y2], where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner. Normalized to 1000x1000.
        label: The name or label of the object in the specified bounding box (optional).
    """
    try:
       kif image_path.startswith("http"):
            response = requests.get(image_path)
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content))
        else:
            image = Image.open(image_path)

        img_width, img_height = image.size

        # bbox is [x1, y1, x2, y2] normalized to 1000x1000
        rel_x1, rel_y1, rel_x2, rel_y2 = bbox_2d
        abs_x1, abs_y1, abs_x2, abs_y2 = rel_x1 / 1000.0 * img_width, rel_y1 / 1000.0 * img_height, rel_x2 / 1000.0 * img_width, rel_y2 / 1000.0 * img_height

        validated_bbox = maybe_resize_bbox(abs_x1, abs_y1, abs_x2, abs_y2, img_width, img_height)
        left, top, right, bottom = validated_bbox

        # Crop the image
        cropped_image = image.crop((left, top, right, bottom))

        # Resize according to smart_resize logic
        new_w, new_h = smart_resize((right - left), (bottom - top), factor=32, min_pixels=256 * 32 * 32)
        cropped_image = cropped_image.resize((new_w, new_h), resample=Image.BICUBIC)

        image_bytes = io.BytesIO()
        # Save as PNG to preserve quality/transparency if any
        cropped_image.save(image_bytes, format="PNG")
        image_bytes.seek(0)

        png_data = base64.b64encode(image_bytes.getvalue()).decode("utf-8")

        return ImageContent(type="image", data=png_data, mimeType="image/png")

    except Exception as e:
        # In a real MCP server, you might want to return an error string or handle it gracefully
        return f"Error processing image: {str(e)}"


if __name__ == "__main__":
    app.run()
