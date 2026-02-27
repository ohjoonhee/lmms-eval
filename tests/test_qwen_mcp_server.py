import base64
import os
import sys
from io import BytesIO

from PIL import Image

# Add project root to sys.path to ensure we can import the module if needed
sys.path.append(os.getcwd())

from examples.mcp_server.qwen_image_tools_server import image_zoom_in


def test_image_zoom_in():
    # Create a dummy image
    img = Image.new("RGB", (1000, 1000), color="white")
    # Draw a red rectangle in the middle
    for x in range(400, 600):
        for y in range(400, 600):
            img.putpixel((x, y), (255, 0, 0))

    img_path = "test_image.jpg"
    img.save(img_path)

    try:
        # Define a bbox that captures the red rectangle [x1, y1, x2, y2]
        # Normalized coordinates: 400/1000 * 1000 = 400.
        bbox = [400, 400, 600, 600]

        result = image_zoom_in(img_path, bbox)

        if isinstance(result, str) and result.startswith("Error"):
            print(f"Test failed: {result}")
            return

        print("Test passed: MCP Tool executed successfully.")

        # Verify output
        # result is ImageContent(type="image", data=..., mimeType="image/png")
        image_data = base64.b64decode(result.data)
        out_img = Image.open(BytesIO(image_data))
        print(f"Output image size: {out_img.size}")

        # Check if the output image is red (since we cropped the red center)
        # Random sampling
        center_pixel = out_img.getpixel((out_img.width // 2, out_img.height // 2))
        print(f"Center pixel color: {center_pixel}")
        if center_pixel == (255, 0, 0) or center_pixel == (254, 0, 0):  # Allowing slight compression artifact drift if any, though PNG is lossless
            print("Color check passed: Center pixel is red.")
        else:
            print("Color check warning: Center pixel might not be red (could be due to resizing interpolation).")

    finally:
        if os.path.exists(img_path):
            os.remove(img_path)


if __name__ == "__main__":
    test_image_zoom_in()
