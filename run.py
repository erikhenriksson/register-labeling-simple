import io
import json
import os
import sys
from typing import List, Dict

import torch
import torch.nn.functional as F
import transformers
import zstandard as zstd
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Set environment variables for HuggingFace
os.environ["HF_HOME"] = ".hf/hf_home"
os.environ["XDG_CACHE_HOME"] = ".hf/xdg_cache_home"


def setup_model_and_tokenizer(model_path: str, batch_size: int = 64) -> tuple:
    """
    Initialize the model and tokenizer.

    Args:
        model_path: Path to the local model
        batch_size: Batch size for processing

    Returns:
        tuple: (model, tokenizer, device, batch_size)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")

    # Print model configuration
    print("Model labels:", model.config.id2label)
    print(f"Number of labels: {len(model.config.id2label)}")

    # Move model to device
    model = model.to(device)
    model.eval()  # Set to evaluation mode

    return model, tokenizer, device, batch_size


def read_zst_jsonl(filepath: str) -> Dict:
    """
    Read compressed JSONL file.

    Args:
        filepath: Path to the .jsonl.zst file

    Yields:
        dict: Parsed JSON object
    """
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


def process_batch(texts: List[str], model, tokenizer, device) -> List[List[float]]:
    """
    Process a batch of texts through the model.

    Args:
        texts: List of input texts
        model: The model
        tokenizer: The tokenizer
        device: The device to use

    Returns:
        List[List[float]]: List of probability lists for each text
    """
    # Tokenize
    inputs = tokenizer(
        texts, padding=True, truncation=True, max_length=2048, return_tensors="pt"
    )

    # Move inputs to device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = F.sigmoid(outputs.logits)

    # Convert to Python list and round values
    return [
        [round(float(prob), 4) for prob in probs]
        for probs in probabilities.cpu().numpy()
    ]


def process_file(
    input_file: str, output_file: str, model, tokenizer, device, batch_size: int
):
    """
    Process the input file and write results to output file.

    Args:
        input_file: Path to input .jsonl.zst file
        output_file: Path to output .jsonl file
        model: The model
        tokenizer: The tokenizer
        device: The device to use
        batch_size: Batch size for processing
    """
    texts = []
    items = []
    total_processed = 0

    with open(output_file, "w", encoding="utf-8") as out_f:
        # Create progress bar for the file reading
        for item in tqdm(read_zst_jsonl(input_file), desc="Processing"):
            texts.append(item["text"])
            items.append(item)

            # Process when we have enough items to fill a batch
            if len(texts) >= batch_size:
                probabilities = process_batch(texts, model, tokenizer, device)

                # Add predictions to items and write to file
                for item, probs in zip(items, probabilities):
                    item["register_probabilities"] = probs
                    out_f.write(json.dumps(item, ensure_ascii=False) + "\n")

                total_processed += len(texts)
                print(f"Processed batch. Total items processed: {total_processed}")
                texts = []
                items = []

        # Process remaining items
        if texts:
            probabilities = process_batch(texts, model, tokenizer, device)

            for item, probs in zip(items, probabilities):
                item["register_probabilities"] = probs
                out_f.write(json.dumps(item, ensure_ascii=False) + "\n")

            total_processed += len(texts)
            print(f"Processed final batch. Total items processed: {total_processed}")


def main():
    if len(sys.argv) != 3:
        print("Usage: python script.py input.jsonl.zst output.jsonl")
        sys.exit(1)

    # Setup
    model_path = "/scratch/project_2011770/bge-2048"
    model, tokenizer, device, batch_size = setup_model_and_tokenizer(model_path)

    # Process file
    process_file(
        input_file=sys.argv[1],
        output_file=sys.argv[2],
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=batch_size,
    )


if __name__ == "__main__":
    main()
