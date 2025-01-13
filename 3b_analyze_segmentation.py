import json
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from collections import defaultdict
import argparse

# Label structure as provided
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


def get_register_label(probs, threshold=0.4):
    """Get single register label from probabilities"""
    labels = process_probabilities(probs, threshold)
    active_indices = np.where(labels == 1)[0]
    if len(active_indices) == 1:
        return labels_all[active_indices[0]]
    return None


def analyze_embeddings(
    data, level="document", min_samples=50, normalize_by_length=True
):
    """Analyze embeddings at document or segment level with optional length normalization"""
    register_embeddings = defaultdict(list)
    register_lengths = defaultdict(list)  # To store text lengths

    for item in data:
        if level == "document":
            probs = item["register_probabilities"]
            emb = item["embedding"]
            text_length = len(item["text"].split())  # Approximate by word count
            register = get_register_label(probs)
            if register:
                register_embeddings[register].append(emb)
                register_lengths[register].append(text_length)
        else:  # segment level
            for probs, emb, text in zip(
                item["segmentation"]["register_probabilities"],
                item["segmentation"]["embeddings"],
                item["segmentation"]["texts"],
            ):
                register = get_register_label(probs)
                if register:
                    register_embeddings[register].append(emb)
                    register_lengths[register].append(len(text.split()))

    # Only keep registers with enough samples
    valid_registers = set(
        k for k, v in register_embeddings.items() if len(v) >= min_samples
    )
    register_embeddings = {
        k: v for k, v in register_embeddings.items() if k in valid_registers
    }
    register_lengths = {
        k: v for k, v in register_lengths.items() if k in valid_registers
    }

    register_variances = {}
    for register, embeddings in register_embeddings.items():
        embeddings_array = np.array(embeddings)
        pca = PCA(n_components=50)
        pca_result = pca.fit_transform(embeddings_array)
        variances = np.var(pca_result, axis=0)
        mean_variance = np.mean(variances)

        if normalize_by_length:
            mean_length = np.mean(register_lengths[register])
            mean_variance = mean_variance / np.sqrt(mean_length)

        register_variances[register] = mean_variance

    return register_variances


def plot_comparative_variances(doc_variances, segment_variances, output_path):
    # Find common registers with enough data
    common_registers = sorted(set(doc_variances.keys()) & set(segment_variances.keys()))

    if not common_registers:
        print(
            "No registers have enough data for both document and segment level analysis"
        )
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    # Set the positions of the bars
    x = np.arange(len(common_registers))
    width = 0.35

    # Create bars
    rects1 = ax.bar(
        x - width / 2,
        [doc_variances[r] for r in common_registers],
        width,
        label="Document Level",
    )
    rects2 = ax.bar(
        x + width / 2,
        [segment_variances[r] for r in common_registers],
        width,
        label="Segment Level",
    )

    # Customize the plot
    ax.set_ylabel("Average Variance (first 50 PCA components)")
    ax.set_title("Embedding Variance Comparison: Document vs Segment Level")
    ax.set_xticks(x)
    ax.set_xticklabels(common_registers, rotation=45)
    ax.legend()

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(f"{output_path}_comparison.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze register embeddings from JSONL files"
    )
    parser.add_argument("files", nargs="+", help="Input JSONL files")
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output path for plots (without extension)",
    )
    args = parser.parse_args()

    # Read data
    data = []
    for file_path in args.files:
        with open(file_path, "r") as f:
            for line in f:
                data.append(json.loads(line))

    # Analyze at both levels
    doc_variances = analyze_embeddings(data, level="document")
    segment_variances = analyze_embeddings(data, level="segment")

    # Print results
    print("\nDocument-level register variances:")
    for register, variance in sorted(doc_variances.items()):
        print(f"{register}: {variance:.4f}")

    print("\nSegment-level register variances:")
    for register, variance in sorted(segment_variances.items()):
        print(f"{register}: {variance:.4f}")

    # Create comparative plot
    plot_comparative_variances(doc_variances, segment_variances, args.output)


if __name__ == "__main__":
    main()
