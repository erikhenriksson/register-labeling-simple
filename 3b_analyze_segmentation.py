import json
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from collections import defaultdict
import argparse
from collections import Counter

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


def analyze_label_frequencies(data):
    """Analyze frequencies of single-label texts for documents and segments"""
    doc_labels = []
    segment_labels = []

    # For tracking multiple labels and no labels
    doc_multi = 0
    doc_none = 0
    seg_multi = 0
    seg_none = 0

    for item in data:
        # Document level
        doc_probs = item["register_probabilities"]
        labels = process_probabilities(doc_probs)
        active = np.where(labels == 1)[0]
        if len(active) == 0:
            doc_none += 1
        elif len(active) == 1:
            doc_labels.append(labels_all[active[0]])
        else:
            doc_multi += 1
            print(f"Multi-label doc example: {[labels_all[i] for i in active]}")

        # Segment level
        for probs in item["segmentation"]["register_probabilities"]:
            labels = process_probabilities(probs)
            active = np.where(labels == 1)[0]
            if len(active) == 0:
                seg_none += 1
            elif len(active) == 1:
                segment_labels.append(labels_all[active[0]])
            else:
                seg_multi += 1
                print(f"Multi-label segment example: {[labels_all[i] for i in active]}")

    # Count frequencies
    print("\nDocument Level Analysis:")
    print(f"Total documents: {len(data)}")
    print(f"No label: {doc_none}")
    print(f"Single label: {len(doc_labels)}")
    print(f"Multiple labels: {doc_multi}")
    print("\nLabel frequencies:")
    doc_freq = Counter(doc_labels)
    for label in sorted(doc_freq.keys()):
        print(f"{label}: {doc_freq[label]}")

    print("\nSegment Level Analysis:")
    total_segments = sum(
        len(item["segmentation"]["register_probabilities"]) for item in data
    )
    print(f"Total segments: {total_segments}")
    print(f"No label: {seg_none}")
    print(f"Single label: {len(segment_labels)}")
    print(f"Multiple labels: {seg_multi}")
    print("\nLabel frequencies:")
    seg_freq = Counter(segment_labels)
    for label in sorted(seg_freq.keys()):
        print(f"{label}: {seg_freq[label]}")


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
    data, level="document", min_samples=50, normalize_by_length=True, doc_lengths=None
):
    """Analyze embeddings at document or segment level with optional length normalization"""
    register_embeddings = defaultdict(list)
    register_lengths = defaultdict(list)

    for item in data:
        if level == "document":
            probs = item["register_probabilities"]
            emb = item["embedding"]
            text_length = len(item["text"].split())
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

        if normalize_by_length and level == "segment" and doc_lengths:
            # Only normalize if we have document length data for this register
            if register in doc_lengths:
                mean_length = np.mean(register_lengths[register])
                doc_mean_length = np.mean(doc_lengths[register])
                mean_variance = mean_variance * np.sqrt(mean_length / doc_mean_length)
            else:
                print(
                    f"Warning: No document-level data for register '{register}', skipping length normalization"
                )

        register_variances[register] = mean_variance

    return register_variances, register_lengths


def plot_comparative_variances(
    doc_variances, segment_variances, output_path, normalized=False
):
    # Find common registers
    common_registers = sorted(set(doc_variances.keys()) & set(segment_variances.keys()))

    if not common_registers:
        print(
            "No registers have enough data for both document and segment level analysis"
        )
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(common_registers))
    width = 0.35

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

    ax.set_ylabel("Average Variance (first 50 PCA components)")
    title = "Embedding Variance Comparison: Document vs Segment Level"
    if normalized:
        title += " (Length Normalized)"
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(common_registers, rotation=45)
    ax.legend()

    plt.tight_layout()
    plt.savefig(f"{output_path}_comparison{'_normalized' if normalized else ''}.png")
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
    parser.add_argument(
        "--normalize",
        "-n",
        action="store_true",
        help="Normalize variances by text length",
    )
    args = parser.parse_args()

    # Read data
    data = []
    for file_path in args.files:
        with open(file_path, "r") as f:
            for line in f:
                data.append(json.loads(line))

    print("Analyzing label frequencies...")
    analyze_label_frequencies(data)

    # First get document level variances and lengths
    doc_variances, doc_lengths = analyze_embeddings(
        data, level="document", normalize_by_length=False
    )

    # Then get segment level variances, using document lengths for normalization
    segment_variances, segment_lengths = analyze_embeddings(
        data,
        level="segment",
        normalize_by_length=args.normalize,
        doc_lengths=doc_lengths,
    )

    # Print results with length information
    print("\nDocument-level analysis:")
    for register in sorted(doc_variances.keys()):
        avg_length = np.mean(doc_lengths[register])
        print(
            f"{register}: variance = {doc_variances[register]:.4f}, avg length = {avg_length:.1f}"
        )

    print("\nSegment-level analysis:")
    for register in sorted(segment_variances.keys()):
        avg_length = np.mean(segment_lengths[register])
        print(
            f"{register}: variance = {segment_variances[register]:.4f}, avg length = {avg_length:.1f}"
        )

    # Create comparative plot
    plot_comparative_variances(
        doc_variances, segment_variances, args.output, args.normalize
    )


if __name__ == "__main__":
    main()
