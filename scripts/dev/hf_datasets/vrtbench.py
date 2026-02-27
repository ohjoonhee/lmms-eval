import io

import tensorflow as tf
from datasets import Dataset, Features, Image, Sequence, Value
from PIL import Image as PILImage


def parse_tfrecord(tfrecord_path: str) -> list[dict]:
    """Parse all examples from a TFRecord file into a list of dicts."""
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    records = []

    for raw_record in dataset:
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        features = example.features.feature

        # Decode image bytes to PIL
        image_bytes = features["image"].bytes_list.value[0]
        image = PILImage.open(io.BytesIO(image_bytes))

        record = {
            "key": features["key"].bytes_list.value[0].decode("utf-8"),
            "image": image,
            "question": features["question"].bytes_list.value[0].decode("utf-8"),
            "class_ids": [v.decode("utf-8") for v in features["class_ids"].bytes_list.value],
            "human_answer_caption": features["human_answer_caption"].bytes_list.value[0].decode("utf-8"),
            "human_reasoning": features["human_reasoning"].bytes_list.value[0].decode("utf-8"),
            "human_confidence": features["human_confidence"].int64_list.value[0],
            "human_labeled_a_objs": list(features["human_labeled_a_objs"].int64_list.value),
            "human_labeled_r_objs": list(features["human_labeled_r_objs"].int64_list.value),
            "objects_info_json": features["objects_info_json"].bytes_list.value[0].decode("utf-8"),
        }
        records.append(record)

    return records


def main() -> None:
    tfrecord_path = "data/vrtbench/vrt_eval.tfrecord"
    print(f"Loading TFRecord from {tfrecord_path}...")
    records = parse_tfrecord(tfrecord_path)
    print(f"Parsed {len(records)} records.")

    hf_features = Features(
        {
            "key": Value("string"),
            "image": Image(),
            "question": Value("string"),
            "class_ids": Sequence(Value("string")),
            "human_answer_caption": Value("string"),
            "human_reasoning": Value("string"),
            "human_confidence": Value("int64"),
            "human_labeled_a_objs": Sequence(Value("int64")),
            "human_labeled_r_objs": Sequence(Value("int64")),
            "objects_info_json": Value("string"),
        }
    )

    ds = Dataset.from_list(records, features=hf_features)
    print(ds)
    print(ds[0])

    output_path = "data/vrtbench/hf_dataset"
    ds.save_to_disk(output_path)
    print(f"Saved HuggingFace dataset to {output_path}")


if __name__ == "__main__":
    main()
