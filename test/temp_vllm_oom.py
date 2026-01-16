import os
import json
import torch
from vllm import LLM, SamplingParams

# Configuration
MODEL_PATH = "Qwen/Qwen3-VL-8B-Thinking"
START_LENGTH = 1000
STEP_LENGTH = 2000
MAX_LENGTH = 65536  # Set a high limit to try to induce OOM before length limit
OUTPUT_FILE = "oom_test_log.json"

# vLLM Configuration
# matching some args observed in scripts, but allowing larger max_model_len
GPU_MEMORY_UTILIZATION = 0.9
ENFORCE_EAGER = False


def run_oom_test():
    print(f"Instantiating model: {MODEL_PATH}")
    print(f"Config: GPU util={GPU_MEMORY_UTILIZATION}, Eager={ENFORCE_EAGER}, Max Len={MAX_LENGTH}")

    try:
        llm = LLM(
            model=MODEL_PATH,
            trust_remote_code=True,
            dtype="bfloat16",
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            max_model_len=MAX_LENGTH,
            enforce_eager=ENFORCE_EAGER,
        )
    except Exception as e:
        print(f"Failed to initialize LLM: {e}")
        return

    sampling_params = SamplingParams(max_tokens=10, temperature=0.7)
    tokenizer = llm.get_tokenizer()
    # Use a generic token id (e.g., 100) if pad_token_id is None
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 100

    results = []
    oom_occurred = False
    oom_length = -1

    current_length = START_LENGTH

    print("Starting iteration...")
    while current_length <= MAX_LENGTH:
        print(f"Testing input lengh: {current_length} tokens...")

        # Construct meaningful prompt
        base_text = "The quick brown fox jumps over the lazy dog. "
        base_tokens = tokenizer.encode(base_text)

        # calculate interactions needed
        prompt_token_ids = []
        while len(prompt_token_ids) < current_length:
            prompt_token_ids.extend(base_tokens)

        # truncate
        prompt_token_ids = prompt_token_ids[:current_length]

        try:
            # Generate
            outputs = llm.generate(prompts=[{"prompt_token_ids": prompt_token_ids}], sampling_params=sampling_params, use_tqdm=False)
            print(f"Length {current_length}: Success")
            results.append({"length": current_length, "status": "success"})

        except torch.cuda.OutOfMemoryError:
            print(f"Length {current_length}: CUDA OOM Error")
            oom_occurred = True
            oom_length = current_length
            results.append({"length": current_length, "status": "oom_cuda"})
            break

        except Exception as e:
            error_msg = str(e)
            if "out of memory" in error_msg.lower():
                print(f"Length {current_length}: OOM detected in exception: {error_msg}")
                oom_occurred = True
                oom_length = current_length
                results.append({"length": current_length, "status": "oom_exception", "message": error_msg})
                break
            else:
                print(f"Length {current_length}: Other error: {error_msg}")
                results.append({"length": current_length, "status": "error", "message": error_msg})
                # If it's a length limit error, we stop but don't count it as OOM
                if "too long" in error_msg:
                    break
                # For other errors, we might want to stop as well
                break

        current_length += STEP_LENGTH

    # Log header
    final_status = {"model": MODEL_PATH, "oom_occurred": oom_occurred, "oom_length": oom_length if oom_occurred else None, "max_tested_length": current_length, "details": results}

    # Save to file
    try:
        with open(OUTPUT_FILE, "w") as f:
            json.dump(final_status, f, indent=4)
        print(f"Results saved to {OUTPUT_FILE}")
    except Exception as e:
        print(f"Failed to save results: {e}")


if __name__ == "__main__":
    run_oom_test()
