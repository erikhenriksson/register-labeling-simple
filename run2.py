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


def read_zst_jsonl(filepath: str, max_lines: int = 10000) -> Dict:
    """
    Read compressed JSONL file with line limit.
    """
    line_count = 0
    with open(filepath, "rb") as fh:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(fh) as reader:
            text_stream = io.TextIOWrapper(reader, encoding="utf-8")
            for line in text_stream:
                if line_count >= max_lines:
                    break
                try:
                    yield json.loads(line.strip())
                    line_count += 1
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}")
                    continue


def process_batch(texts: List[str], model, tokenizer, device) -> tuple:
    """
    Process a batch of texts through the model, returning both probabilities and embeddings.
    """
    # Tokenize
    inputs = tokenizer(
        texts, padding=True, truncation=True, max_length=2048, return_tensors="pt"
    )

    # Move inputs to device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Get predictions and embeddings
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = F.sigmoid(outputs.logits)
        # Get embeddings from the last hidden state
        embeddings = (
            outputs.hidden_states[-1][:, 0, :]
            if hasattr(outputs, "hidden_states")
            else None
        )

    # Convert probabilities to Python list and round values
    prob_list = [
        [round(float(prob), 4) for prob in probs]
        for probs in probabilities.cpu().numpy()
    ]

    # Convert embeddings to list of floats if available
    emb_list = (
        [[float(x) for x in emb] for emb in embeddings.cpu().numpy()]
        if embeddings is not None
        else [[] for _ in texts]
    )

    return prob_list, emb_list


def process_file(
    input_file: str, output_file: str, model, tokenizer, device, batch_size: int
):
    """
    Process the input file and write results to output file.
    """
    texts = []
    total_processed = 0

    with open(output_file, "w", encoding="utf-8") as out_f:
        # Create progress bar for the file reading
        for item in tqdm(read_zst_jsonl(input_file), desc="Processing"):
            texts.append(item["text"])

            # Process when we have enough items to fill a batch
            if len(texts) >= batch_size:
                probabilities, embeddings = process_batch(
                    texts, model, tokenizer, device
                )

                # Write truncated results to file
                for text, probs, emb in zip(texts, probabilities, embeddings):
                    # Truncate text using tokenizer
                    truncated_text = tokenizer.decode(
                        tokenizer.encode(text, max_length=2048, truncation=True),
                        skip_special_tokens=True,
                    )

                    output_item = {
                        "text": truncated_text,
                        "register_probabilities": probs,
                        "embedding": emb,
                    }
                    out_f.write(json.dumps(output_item, ensure_ascii=False) + "\n")
                    out_f.flush()  # Flush after each batch

                total_processed += len(texts)
                print(f"Processed batch. Total items processed: {total_processed}")
                texts = []

        # Process remaining items
        if texts:
            probabilities, embeddings = process_batch(texts, model, tokenizer, device)

            for text, probs, emb in zip(texts, probabilities, embeddings):
                truncated_text = tokenizer.decode(
                    tokenizer.encode(text, max_length=2048, truncation=True),
                    skip_special_tokens=True,
                )

                output_item = {
                    "text": truncated_text,
                    "register_probabilities": probs,
                    "embedding": emb,
                }
                out_f.write(json.dumps(output_item, ensure_ascii=False) + "\n")
                out_f.flush()

            total_processed += len(texts)
            print(f"Processed final batch. Total items processed: {total_processed}")


def main():
    if len(sys.argv) != 3:
        print("Usage: python script.py input.jsonl.zst output.jsonl")
        sys.exit(1)

    # Setup with output_hidden_states=True to get embeddings
    model_path = "/scratch/project_2011770/bge-2048"
    model, tokenizer, device, batch_size = setup_model_and_tokenizer(model_path)

    # Enable hidden states output
    model.config.output_hidden_states = True

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
