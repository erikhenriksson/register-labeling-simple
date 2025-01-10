import json
import os
from typing import Dict, List
import argparse

# Label structure definition
LABELS_STRUCTURE = {
    "MT": [],
    "LY": [],
    "SP": ["it"],
    "ID": [],
    "NA": ["ne", "sr", "nb"],
    "HI": ["re"],
    "IN": ["en", "ra", "dtp", "fi", "lt"],
    "OP": ["rv", "ob", "rs", "av"],
    "IP": ["ds", "ed"],
}

# Create flat list of labels
LABELS_LIST = [k for k in LABELS_STRUCTURE.keys()] + [
    item for row in LABELS_STRUCTURE.values() for item in row
]


def get_parent_label(child_label: str) -> str:
    """Find the parent label for a given child label."""
    for parent, children in LABELS_STRUCTURE.items():
        if child_label in children:
            return parent
    return None


def process_probabilities(probabilities: List[float], threshold: float) -> List[str]:
    """
    Process probabilities and return active labels, implementing the parent-child logic.
    If a child label is active, its parent label is deactivated.
    """
    # Create dictionary of label:probability pairs
    label_probs = dict(zip(LABELS_LIST, probabilities))

    # First pass: identify all labels above threshold
    active_labels = {label for label, prob in label_probs.items() if prob >= threshold}

    # Second pass: handle parent-child relationships
    final_labels = set()
    for label in active_labels:
        parent = get_parent_label(label)
        if parent:
            # If this is a child label, add it and remove its parent if present
            final_labels.add(label)
            active_labels.discard(parent)
        else:
            # If this is a parent label, only add it if none of its children are active
            children = set(LABELS_STRUCTURE[label])
            if not children.intersection(active_labels):
                final_labels.add(label)

    return sorted(list(final_labels))


def process_file(input_file: str, threshold: float):
    """Process the input JSONL file and create output directories and files."""
    base_dir = os.path.dirname(input_file)
    filename = os.path.basename(input_file)

    # Create root splits directory
    splits_dir = os.path.join(base_dir, "splits")
    os.makedirs(splits_dir, exist_ok=True)

    with open(input_file, "r") as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                probabilities = data["register_probabilities"]

                # Get active labels
                active_labels = process_probabilities(probabilities, threshold)

                # Create directory name from sorted, comma-separated labels, or "none" if no labels
                dir_name = ",".join(sorted(active_labels)) if active_labels else "none"

                # Create directory under splits directory
                output_dir = os.path.join(splits_dir, dir_name)
                os.makedirs(output_dir, exist_ok=True)

                # Write to output file
                output_file = os.path.join(output_dir, filename)
                with open(output_file, "a") as out_f:
                    out_f.write(line)

            except json.JSONDecodeError as e:
                print(f"Error processing line: {e}")
            except KeyError as e:
                print(f"Missing key in JSON: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Process JSONL file with hierarchical labels."
    )
    parser.add_argument("input_file", help="Input JSONL file path")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.4,
        help="Probability threshold for label activation (default: 0.4)",
    )

    args = parser.parse_args()

    process_file(args.input_file, args.threshold)


if __name__ == "__main__":
    main()
