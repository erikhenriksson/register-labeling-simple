import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder


def compute_mutual_information(embeddings, labels):
    """
    Compute mutual information between embeddings and class labels.

    Args:
        embeddings: np.array of shape (n_samples, n_features)
        labels: list of string labels

    Returns:
        float: average MI across features
        np.array: MI for each feature
    """
    # Convert string labels to integers
    le = LabelEncoder()
    label_ids = le.fit_transform(labels)

    # Compute MI for each feature
    mi_scores = mutual_info_classif(embeddings, label_ids, discrete_features=False)
    return np.mean(mi_scores), mi_scores


def analyze_embeddings_with_mi(data, level="document", min_samples=10):
    """
    Analyze embeddings using mutual information at document or segment level.
    """
    register_embeddings = defaultdict(list)
    all_embeddings = []
    all_labels = []

    for item in data:
        if level == "document":
            probs = item["register_probabilities"]
            emb = item["embedding"]
            register = get_register_label(probs)
            if register:
                register_embeddings[register].append(emb)
                all_embeddings.append(emb)
                all_labels.append(register)
        else:  # segment level
            for probs, emb in zip(
                item["segmentation"]["register_probabilities"],
                item["segmentation"]["embeddings"],
            ):
                register = get_register_label(probs)
                if register:
                    register_embeddings[register].append(emb)
                    all_embeddings.append(emb)
                    all_labels.append(register)

    # Only keep registers with enough samples
    valid_registers = {
        k for k, v in register_embeddings.items() if len(v) >= min_samples
    }

    # Filter data to only include valid registers
    filtered_embeddings = []
    filtered_labels = []
    for emb, label in zip(all_embeddings, all_labels):
        if label in valid_registers:
            filtered_embeddings.append(emb)
            filtered_labels.append(label)

    if not filtered_embeddings:
        return None, None, {}

    # Convert to numpy array
    embeddings_array = np.array(filtered_embeddings)

    # Compute MI on raw embeddings
    mi_raw, mi_scores_raw = compute_mutual_information(
        embeddings_array, filtered_labels
    )

    # Compute MI on PCA components
    pca = PCA(n_components=10)
    pca_result = pca.fit_transform(embeddings_array)
    mi_pca, mi_scores_pca = compute_mutual_information(pca_result, filtered_labels)

    # Compute per-register MI
    register_mi = {}
    for register in valid_registers:
        register_mask = np.array(filtered_labels) == register
        register_embeddings = embeddings_array[register_mask]
        others_embeddings = embeddings_array[~register_mask]

        # Create binary labels for one-vs-rest MI computation
        binary_labels = [
            "positive" if r == register else "negative" for r in filtered_labels
        ]
        mi_score, _ = compute_mutual_information(embeddings_array, binary_labels)
        register_mi[register] = mi_score

    return mi_raw, mi_pca, register_mi


def plot_mi_comparison(doc_mi, segment_mi, output_path):
    """Plot comparative MI analysis."""
    common_registers = sorted(set(doc_mi.keys()) & set(segment_mi.keys()))

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
        [doc_mi[r] for r in common_registers],
        width,
        label="Document Level",
    )
    rects2 = ax.bar(
        x + width / 2,
        [segment_mi[r] for r in common_registers],
        width,
        label="Segment Level",
    )

    ax.set_ylabel("Mutual Information")
    ax.set_title("Mutual Information Comparison: Document vs Segment Level")
    ax.set_xticks(x)
    ax.set_xticklabels(common_registers, rotation=45)
    ax.legend()

    plt.tight_layout()
    plt.savefig(f"{output_path}_mi_comparison.png")
    plt.close()


# Modify main() to include MI analysis
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

    # Compute MI for documents and segments
    print("\nComputing mutual information...")
    doc_mi_raw, doc_mi_pca, doc_mi_per_register = analyze_embeddings_with_mi(
        data, level="document"
    )
    seg_mi_raw, seg_mi_pca, seg_mi_per_register = analyze_embeddings_with_mi(
        data, level="segment"
    )

    print("\nDocument-level MI analysis:")
    print(f"Raw embeddings MI: {doc_mi_raw:.4f}")
    print(f"PCA components MI: {doc_mi_pca:.4f}")
    print("\nPer-register MI:")
    for register in sorted(doc_mi_per_register.keys()):
        print(f"{register}: MI = {doc_mi_per_register[register]:.4f}")

    print("\nSegment-level MI analysis:")
    print(f"Raw embeddings MI: {seg_mi_raw:.4f}")
    print(f"PCA components MI: {seg_mi_pca:.4f}")
    print("\nPer-register MI:")
    for register in sorted(seg_mi_per_register.keys()):
        print(f"{register}: MI = {seg_mi_per_register[register]:.4f}")

    # Create comparative MI plot
    plot_mi_comparison(doc_mi_per_register, seg_mi_per_register, args.output)

    # Also run the original variance analysis
    doc_variances, doc_lengths = analyze_embeddings(
        data, level="document", normalize_by_length=False
    )
    segment_variances, segment_lengths = analyze_embeddings(
        data,
        level="segment",
        normalize_by_length=args.normalize,
        doc_lengths=doc_lengths,
    )
    plot_comparative_variances(
        doc_variances, segment_variances, args.output, args.normalize
    )
