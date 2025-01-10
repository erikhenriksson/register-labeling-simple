import io
import json
import os
import sys

import transformers
import zstandard as zstd
from tqdm import tqdm
import torch

os.environ["HF_HOME"] = ".hf/hf_home"
os.environ["XDG_CACHE_HOME"] = ".hf/xdg_cache_home"

# Set up pipeline with local model and HF tokenizer
model_path = "/scratch/project_2011770/bge-2048"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipeline = transformers.pipeline(
    task="text-classification",
    model=model_path,
    tokenizer="xlm-roberta-large",
    top_k=None,
    function_to_apply="sigmoid",
    batch_size=64,
    max_length=2048,
    truncation=True,
    device=device,  # Add device argument
)


def read_zst_jsonl(filepath):
    with open(filepath, "rb") as fh:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(fh) as reader:
            text_stream = io.TextIOWrapper(reader, encoding="utf-8")
            for line in text_stream:
                try:
                    yield json.loads(line.strip())
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}")
                    continue


def process_file(input_file, output_file):
    texts = []
    items = []
    total_processed = 0

    with open(output_file, "w", encoding="utf-8") as out_f:
        for item in tqdm(read_zst_jsonl(input_file)):
            texts.append(item["text"])
            items.append(item)

            # Process when we have enough items to fill a batch
            if len(texts) >= pipeline._batch_size:
                results = pipeline(texts)
                for item, preds in zip(items, results):
                    item["register_probabilities"] = [
                        round(float(p["score"]), 4) for p in preds
                    ]
                    out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
                total_processed += len(texts)
                print(f"Processed batch. Total items processed: {total_processed}")
                texts = []
                items = []

        # Process remaining items
        if texts:
            results = pipeline(texts)
            for item, preds in zip(items, results):
                item["register_probabilities"] = [
                    round(float(p["score"]), 4) for p in preds
                ]
                out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
            total_processed += len(texts)
            print(f"Processed final batch. Total items processed: {total_processed}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py input.jsonl.zst output.jsonl")
        sys.exit(1)

    process_file(sys.argv[1], sys.argv[2])
