import asyncio
import json

from datasets import load_from_disk
from openai import AsyncOpenAI
import dotenv

dotenv.load_dotenv()

SIMPLIFY_PROMPT = """\
You are given a detailed object caption from an image. \
Produce a single concise noun phrase (1-5 words) that distinctively \
identifies the object. Focus on what the object IS, not its appearance details.

Output ONLY the noun phrase, nothing else.

Caption:
{caption}"""


async def simplify_caption(
    client: AsyncOpenAI,
    caption: str,
    semaphore: asyncio.Semaphore,
) -> str:
    """Call GPT-4.1-nano to simplify a single caption into a short phrase."""
    async with semaphore:
        response = await client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[{"role": "user", "content": SIMPLIFY_PROMPT.format(caption=caption)}],
            temperature=0,
            max_tokens=64,
        )
    return response.choices[0].message.content.strip()


async def process_dataset(ds):
    """Process all examples concurrently to add simplified answer labels."""
    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(50)

    # Flatten: collect (example_idx, caption) for every answer object
    jobs: list[tuple[int, str]] = []
    for idx, example in enumerate(ds):
        obj_info = json.loads(example["objects_info_json"])
        for oid in example["human_labeled_a_objs"]:
            caption = obj_info.get(str(oid), {}).get("caption", "")
            jobs.append((idx, caption))

    # Fire one request per caption concurrently
    tasks = [simplify_caption(client, caption, semaphore) for _, caption in jobs]
    labels = await asyncio.gather(*tasks)

    # Group results back by example index
    results: list[list[str]] = [[] for _ in range(len(ds))]
    for (idx, _), label in zip(jobs, labels):
        results[idx].append(label)

    return results


def main():
    ds = load_from_disk("data/vrtbench/hf_dataset")
    print(ds)

    simplified = asyncio.run(process_dataset(ds))

    # Also build the full answer_captions column
    def add_columns(example, idx):
        obj_info = json.loads(example["objects_info_json"])
        example["answer_captions"] = [
            obj_info.get(str(oid), {}).get("caption", "")
            for oid in example["human_labeled_a_objs"]
        ]
        example["answer_labels"] = simplified[idx]
        return example

    ds = ds.map(add_columns, with_indices=True)

    # Preview
    for i in range(5):
        print(f"[{i}] Q: {ds[i]['question']}")
        print(f"    captions: {ds[i]['answer_captions']}")
        print(f"    labels:   {ds[i]['answer_labels']}")
        print()

    ds.push_to_hub("ohjoonhee/VRTBench")


if __name__ == "__main__":
    main()
