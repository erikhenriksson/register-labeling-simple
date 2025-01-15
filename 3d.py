import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import argparse
from pathlib import Path

# Your existing label structure
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

labels_all = [k for k in labels_structure.keys()] + [
    item for row in labels_structure.values() for item in row
]


def get_parent_indices(label_structure):
    parent_map = {}
    for parent, children in label_structure.items():
        parent_idx = labels_all.index(parent)
        for child in children:
            child_idx = labels_all.index(child)
            parent_map[child_idx] = parent_idx
    return parent_map


def process_probabilities(probs, threshold=0.4):
    """Convert probabilities to labels considering hierarchy"""
    parent_indices = get_parent_indices(labels_structure)
    labels = (np.array(probs) > threshold).astype(int)

    # Zero out parent when child is active
    for child_idx, parent_idx in parent_indices.items():
        if labels[child_idx] == 1:
            labels[parent_idx] = 0
    return labels


def analyze_registers(input_file, output_file, threshold=0.4):
    # Storage for analysis
    doc_labels = []
    segment_labels = []
    segment_counts = []

    # Read and process data
    with open(input_file, "r") as f:
        for line in f:
            data = json.loads(line)

            # Process document-level labels
            doc_label = process_probabilities(data["register_probabilities"], threshold)
            doc_labels.append(doc_label)

            # Process segment-level labels
            seg_labels = [
                process_probabilities(probs, threshold)
                for probs in data["segmentation"]["register_probabilities"]
            ]
            segment_labels.extend(seg_labels)
            segment_counts.append(len(seg_labels))

    # Convert to numpy arrays
    doc_labels = np.array(doc_labels)
    segment_labels = np.array(segment_labels)

    # Create figure with subplots
    plt.figure(figsize=(20, 15))

    # 1. Distribution Comparison
    plt.subplot(2, 2, 1)
    doc_freq = doc_labels.mean(axis=0)
    seg_freq = segment_labels.mean(axis=0)

    x = np.arange(len(labels_all))
    width = 0.35

    plt.bar(x - width / 2, doc_freq, width, label="Document-level")
    plt.bar(x + width / 2, seg_freq, width, label="Segment-level")
    plt.xticks(x, labels_all, rotation=45)
    plt.title("Register Distribution Comparison")
    plt.legend()

    # 2. Hierarchical Consistency
    plt.subplot(2, 2, 2)
    parent_child_consistency = {}

    for parent, children in labels_structure.items():
        if children:  # if parent has children
            parent_idx = labels_all.index(parent)
            child_indices = [labels_all.index(child) for child in children]

            # Check document level
            doc_consistency = np.mean(
                [
                    (doc_labels[:, parent_idx] == 0)
                    & (doc_labels[:, child_indices].sum(axis=1) > 0)
                ]
            )

            # Check segment level
            seg_consistency = np.mean(
                [
                    (segment_labels[:, parent_idx] == 0)
                    & (segment_labels[:, child_indices].sum(axis=1) > 0)
                ]
            )

            parent_child_consistency[parent] = {
                "document": doc_consistency,
                "segment": seg_consistency,
            }

    parents = list(parent_child_consistency.keys())
    doc_cons = [parent_child_consistency[p]["document"] for p in parents]
    seg_cons = [parent_child_consistency[p]["segment"] for p in parents]

    x = np.arange(len(parents))
    plt.bar(x - width / 2, doc_cons, width, label="Document-level")
    plt.bar(x + width / 2, seg_cons, width, label="Segment-level")
    plt.xticks(x, parents, rotation=45)
    plt.title("Parent-Child Consistency")
    plt.legend()

    # 3. Register Co-occurrence
    plt.subplot(2, 2, 3)

    # Handle correlation calculation with error checking
    def safe_corrcoef(x):
        # Check if we have any variation in the data
        if x.shape[0] <= 1 or not np.any(np.std(x, axis=0) > 0):
            return np.zeros((x.shape[1], x.shape[1]))

        with np.errstate(divide="ignore", invalid="ignore"):
            corr = np.corrcoef(x.T)
            # Replace NaN values with 0
            corr = np.nan_to_num(corr, nan=0.0)
        return corr

    doc_cooc = safe_corrcoef(doc_labels)
    seg_cooc = safe_corrcoef(segment_labels)

    # Plot difference in co-occurrence
    diff_cooc = doc_cooc - seg_cooc
    sns.heatmap(
        diff_cooc,
        xticklabels=labels_all,
        yticklabels=labels_all,
        cmap="RdBu_r",
        center=0,
    )
    plt.title("Co-occurrence Difference (Document - Segment)")

    # 4. Segment Count vs Register Diversity
    plt.subplot(2, 2, 4)
    doc_diversity = np.sum(doc_labels, axis=1)

    # Calculate average segment diversity with error handling
    avg_seg_diversity = []
    cum_counts = np.cumsum([0] + segment_counts[:-1])
    for i, count in enumerate(cum_counts):
        if count >= len(segment_labels):
            # Skip if we're out of bounds
            avg_seg_diversity.append(0)
            continue
        end_idx = min(count + segment_counts[i], len(segment_labels))
        segment_sums = np.sum(segment_labels[count:end_idx], axis=1)
        if len(segment_sums) > 0:
            avg_seg_diversity.append(np.mean(segment_sums))
        else:
            avg_seg_diversity.append(0)

    plt.scatter(segment_counts, doc_diversity, alpha=0.5, label="Document diversity")
    plt.scatter(
        segment_counts, avg_seg_diversity, alpha=0.5, label="Avg segment diversity"
    )
    plt.xlabel("Number of segments")
    plt.ylabel("Number of registers")
    plt.title("Segment Count vs Register Diversity")
    plt.legend()

    # Save the figure
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze register predictions")
    parser.add_argument("input_file", type=str, help="Input JSONL file")
    parser.add_argument("output_file", type=str, help="Output PNG file")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.4,
        help="Probability threshold for binary classification",
    )

    args = parser.parse_args()
    analyze_registers(args.input_file, args.output_file, args.threshold)


if __name__ == "__main__":
    main()
