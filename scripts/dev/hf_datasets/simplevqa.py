import base64
import io

import datasets
from datasets import load_dataset
from PIL import Image

ds = load_dataset("m-a-p/SimpleVQA", split="test")
print(ds)


# ds = ds.filter(lambda example: example["language"] == "EN")
# print("After filtering to EN language:")
# print(ds)


# map
def image_string_to_pil(example):
    try:
        image_data = base64.b64decode(example["image"])
        example["image"] = Image.open(io.BytesIO(image_data)).convert("RGB")
        example["valid_image"] = True
    except (base64.binascii.Error, Image.UnidentifiedImageError, Exception) as e:
        print(f"Warning: Failed to decode image for data_id={example.get('data_id', 'unknown')}: {e}")
        example["image"] = Image.new("RGB", (224, 224), (0, 0, 0))
        example["valid_image"] = False
    return example


ds = ds.map(image_string_to_pil)

ds = ds.cast_column("image", datasets.Image())
print("After converting image strings to PIL images:")
print(ds)

ds = ds.filter(lambda example: example["valid_image"])
ds = ds.remove_columns("valid_image")
print("After filtering out invalid images:")
print(ds)

ds.push_to_hub("SimpleVQA")
