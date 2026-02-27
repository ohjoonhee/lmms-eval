from datasets import load_dataset

# main dataset
ds = load_dataset("duongtr/ReasonVQA", data_files={"train": "train.jsonl", "validation": "val.jsonl"})
print(ds)

# additional annotations
ann = load_dataset("duongtr/ReasonVQA", data_files={"train": "train_ann.jsonl", "validation": "val_ann.jsonl"})
print(ann)
