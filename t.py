import json
from collections import Counter
from typing import Dict, List, Set
import argparse


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
    parent_only: bool = False,
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

    # If parent_only mode is enabled, convert all labels to their parent categories
    if parent_only:
        final_labels = {child_to_parent.get(label, label) for label in final_labels}

    return final_labels


def process_jsonl_file(file_path: str, parent_only: bool = False) -> tuple:
    """
    Process entire JSONL file and return label frequencies and percentages.
    """
    # Setup label structure
    labels_structure, labels_list, child_to_parent = setup_label_structure()

    # Process file
    label_counter = Counter()
    total_records = 0
    hybrid_count = 0
    nonhybrid_count = 0  # includes both single-label and none
    with open(file_path, "r") as f:
        for line in f:
            if total_records > 1000:
                continue
            total_records += 1
            data = json.loads(line)
            probs = data["register_probabilities"]

            # Get labels for this record
            record_labels = process_probabilities(
                probs, labels_list, child_to_parent, parent_only=parent_only
            )

            # Count hybrids vs non-hybrids
            if len(record_labels) > 1:
                hybrid_count += 1
                label_counter["+".join(sorted(record_labels))] += 1
            elif len(record_labels) == 1:
                nonhybrid_count += 1
                label_counter[next(iter(record_labels))] += 1
            else:
                nonhybrid_count += 1
                label_counter["none"] += 1

    # Calculate percentages
    label_percentages = {
        label: (count / total_records) * 100 for label, count in label_counter.items()
    }

    return (
        dict(label_counter),
        label_percentages,
        total_records,
        hybrid_count,
        nonhybrid_count,
    )


if __name__ == "__main__":
    # Setup command line argument parsing
    parser = argparse.ArgumentParser(
        description="Process JSONL file for label analysis."
    )
    parser.add_argument("file_path", help="Path to the JSONL file")
    args = parser.parse_args()

    # Process the file - detailed analysis
    frequencies, percentages, total, hybrid_count, nonhybrid_count = process_jsonl_file(
        args.file_path, parent_only=False
    )

    # Print detailed analysis
    print("\n=== DETAILED ANALYSIS (All Labels) ===")
    print(f"Processed {total} records")
    print(f"Hybrid combinations: {hybrid_count} ({(hybrid_count/total)*100:.2f}%)")
    print(f"Non-hybrid cases: {nonhybrid_count} ({(nonhybrid_count/total)*100:.2f}%)")

    print("\nDetailed Label Frequencies (sorted by frequency):")
    print("-" * 50)
    sorted_items = sorted(frequencies.items(), key=lambda x: (-x[1], x[0]))
    for label, count in sorted_items:
        print(f"{label}: {count} ({percentages[label]:.2f}%)")

    # Process the file - parent categories only
    (
        parent_frequencies,
        parent_percentages,
        parent_total,
        parent_hybrid_count,
        parent_nonhybrid_count,
    ) = process_jsonl_file(args.file_path, parent_only=True)

    # Print parent-level analysis
    print("\n\n=== PARENT CATEGORY ANALYSIS ===")
    print(f"Processed {parent_total} records")
    print(
        f"Hybrid combinations: {parent_hybrid_count} ({(parent_hybrid_count/parent_total)*100:.2f}%)"
    )
    print(
        f"Non-hybrid cases: {parent_nonhybrid_count} ({(parent_nonhybrid_count/parent_total)*100:.2f}%)"
    )

    print("\nParent Category Frequencies (sorted by frequency):")
    print("-" * 50)
    parent_sorted_items = sorted(
        parent_frequencies.items(), key=lambda x: (-x[1], x[0])
    )
    for label, count in parent_sorted_items:
        print(f"{label}: {count} ({parent_percentages[label]:.2f}%)")
