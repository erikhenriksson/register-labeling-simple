import json
from collections import Counter
from typing import Dict, List, Set


def setup_label_structure() -> tuple:
    # Define the label structure
    labels_structure = {
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
    labels_list = [k for k in labels_structure.keys()] + [
        item for row in labels_structure.values() for item in row
    ]

    # Create mapping of child labels to their parents
    child_to_parent = {}
    for parent, children in labels_structure.items():
        for child in children:
            child_to_parent[child] = parent

    return labels_structure, labels_list, child_to_parent


def process_probabilities(
    probs: List[float],
    labels_list: List[str],
    child_to_parent: Dict[str, str],
    threshold: float = 0.4,
) -> Set[str]:
    """
    Process probability list and return labels that pass threshold,
    zeroing out parent categories when children pass threshold.
    """
    # Create dictionary of label:probability pairs
    label_probs = dict(zip(labels_list, probs))

    # First pass: identify which children pass threshold
    children_passed = {}
    for label, prob in label_probs.items():
        if label in child_to_parent and prob >= threshold:
            parent = child_to_parent[label]
            if parent not in children_passed:
                children_passed[parent] = []
            children_passed[parent].append(label)

    # Second pass: zero out parents where children passed threshold
    for parent in children_passed.keys():
        label_probs[parent] = 0

    # Get final labels that pass threshold
    final_labels = {label for label, prob in label_probs.items() if prob >= threshold}

    return final_labels


def process_jsonl_file(file_path: str) -> tuple:
    """
    Process entire JSONL file and return label frequencies and percentages.
    """
    # Setup label structure
    labels_structure, labels_list, child_to_parent = setup_label_structure()

    # Process file
    label_counter = Counter()
    total_records = 0

    with open(file_path, "r") as f:
        for line in f:
            total_records += 1
            data = json.loads(line)
            probs = data["register_probabilities"]

            # Get labels for this record
            record_labels = process_probabilities(probs, labels_list, child_to_parent)

            # If it's a hybrid, count it as a hybrid only
            if len(record_labels) > 1:
                label_counter["+".join(sorted(record_labels))] += 1
            # If single label, count it individually
            elif len(record_labels) == 1:
                label_counter[next(iter(record_labels))] += 1
            # If no labels pass threshold, count as "none"
            else:
                label_counter["none"] += 1

    # Calculate percentages
    label_percentages = {
        label: (count / total_records) * 100 for label, count in label_counter.items()
    }

    return dict(label_counter), label_percentages, total_records


if __name__ == "__main__":
    # Example usage
    file_path = "fin_output.jsonl"
    frequencies, percentages, total = process_jsonl_file(file_path)

    print(f"\nProcessed {total} records")
    print("\nLabel Frequencies (sorted by frequency):")
    # Sort by frequency (count) in descending order
    sorted_items = sorted(frequencies.items(), key=lambda x: (-x[1], x[0]))
    for label, count in sorted_items:
        print(f"{label}: {count} ({percentages[label]:.2f}%)")
