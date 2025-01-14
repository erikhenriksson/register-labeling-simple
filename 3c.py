import json
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from collections import defaultdict
import argparse
from collections import Counter
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder

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


def binary_entropy(p):
    """Compute binary entropy."""
    if p == 0 or p == 1:
        return 0
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def compute_register_entropies(data, level="document"):
    """
    Compute label entropy and embedding entropy for each register.

    Returns:
        dict: Register -> (label_entropy, embedding_entropy_raw, embedding_entropy_pca)
    """
    register_data = defaultdict(lambda: {"embeddings": [], "is_register": []})

    # Collect data
    for item in data:
        if level == "document":
            probs = item["register_probabilities"]
            emb = item["embedding"]
            register = get_register_label(probs)
            if register:
                # Add to positive examples for this register
                register_data[register]["embeddings"].append(emb)
                register_data[register]["is_register"].append(1)
                # Add to negative examples for all other registers
                for other_reg in register_data.keys():
                    if other_reg != register:
                        register_data[other_reg]["embeddings"].append(emb)
                        register_data[other_reg]["is_register"].append(0)
        else:  # segment level
            for probs, emb in zip(
                item["segmentation"]["register_probabilities"],
                item["segmentation"]["embeddings"],
            ):
                register = get_register_label(probs)
                if register:
                    register_data[register]["embeddings"].append(emb)
                    register_data[register]["is_register"].append(1)
                    for other_reg in register_data.keys():
                        if other_reg != register:
                            register_data[other_reg]["embeddings"].append(emb)
                            register_data[other_reg]["is_register"].append(0)

    # Compute entropies
    register_entropies = {}
    for register, data in register_data.items():
        if len(data["embeddings"]) < 10:  # Skip registers with too few samples
            continue

        # Label entropy
        p_register = np.mean(data["is_register"])
        label_entropy = binary_entropy(p_register)

        # Embedding entropy (raw)
        embeddings = np.array(data["embeddings"])
        labels = np.array(data["is_register"])

        # For embedding entropy, we'll use a simple logistic regression
        # and look at the entropy of its predictions
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_predict

        # Raw embeddings
        clf = LogisticRegression(max_iter=1000)
        preds_raw = cross_val_predict(
            clf, embeddings, labels, cv=5, method="predict_proba"
        )
        embedding_entropy_raw = np.mean([binary_entropy(p[1]) for p in preds_raw])

        # PCA embeddings
        pca = PCA(n_components=10)
        embeddings_pca = pca.fit_transform(embeddings)
        preds_pca = cross_val_predict(
            clf, embeddings_pca, labels, cv=5, method="predict_proba"
        )
        embedding_entropy_pca = np.mean([binary_entropy(p[1]) for p in preds_pca])

        register_entropies[register] = (
            label_entropy,
            embedding_entropy_raw,
            embedding_entropy_pca,
        )

    return register_entropies


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


def analyze_embeddings(
    data, level="document", min_samples=10, normalize_by_length=True, doc_lengths=None
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
        pca = PCA(n_components=10)
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

    ax.set_ylabel("Average Variance (first 10 PCA components)")
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


def plot_entropy_comparison(doc_entropies, segment_entropies, output_path):
    """Plot entropy comparisons between documents and segments."""
    common_registers = sorted(set(doc_entropies.keys()) & set(segment_entropies.keys()))

    if not common_registers:
        print(
            "No registers have enough data for both document and segment level analysis"
        )
        return

    x = np.arange(len(common_registers))
    width = 0.35

    # Create single plot for prediction entropies
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(
        x - width / 2,
        [doc_entropies[r] for r in common_registers],
        width,
        label="Document Level",
    )
    ax.bar(
        x + width / 2,
        [segment_entropies[r] for r in common_registers],
        width,
        label="Segment Level",
    )

    ax.set_ylabel("Prediction Entropy")
    ax.set_title("Prediction Entropy Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(common_registers, rotation=45)
    ax.legend()

    plt.tight_layout()
    plt.savefig(f"{output_path}_entropy_comparison.png")
    plt.close()


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


def compute_prediction_entropies(data, level="document"):
    """
    Compute average entropy of predictions for each register.
    """
    register_entropies = {}

    for item in data:
        if level == "document":
            probs = item["register_probabilities"]
            register = get_register_label(probs)
            if register:
                # Get binary entropy for each register prediction
                for reg_idx, reg_prob in enumerate(probs):
                    reg_name = labels_all[reg_idx]
                    if reg_name not in register_entropies:
                        register_entropies[reg_name] = []
                    register_entropies[reg_name].append(binary_entropy(reg_prob))
        else:  # segment level
            for probs in item["segmentation"]["register_probabilities"]:
                register = get_register_label(probs)
                if register:
                    for reg_idx, reg_prob in enumerate(probs):
                        reg_name = labels_all[reg_idx]
                        if reg_name not in register_entropies:
                            register_entropies[reg_name] = []
                        register_entropies[reg_name].append(binary_entropy(reg_prob))

    # Compute average entropy for each register
    return {
        reg: np.mean(entropies)
        for reg, entropies in register_entropies.items()
        if len(entropies) >= 10
    }  # Only return registers with enough samples


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

    # Compute prediction entropies
    print("\nComputing prediction entropies...")
    doc_entropies = compute_prediction_entropies(data, level="document")
    seg_entropies = compute_prediction_entropies(data, level="segment")

    print("\nDocument-level prediction entropy:")
    for register in sorted(doc_entropies.keys()):
        print(f"{register}: {doc_entropies[register]:.4f}")

    print("\nSegment-level prediction entropy:")
    for register in sorted(seg_entropies.keys()):
        print(f"{register}: {seg_entropies[register]:.4f}")

    # Create entropy comparison plot
    plot_entropy_comparison(doc_entropies, seg_entropies, args.output)

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

    # Print variance results with length information
    print("\nDocument-level variance analysis:")
    for register in sorted(doc_variances.keys()):
        avg_length = np.mean(doc_lengths[register])
        print(
            f"{register}: variance = {doc_variances[register]:.4f}, avg length = {avg_length:.1f}"
        )

    print("\nSegment-level variance analysis:")
    for register in sorted(segment_variances.keys()):
        avg_length = np.mean(segment_lengths[register])
        print(
            f"{register}: variance = {segment_variances[register]:.4f}, avg length = {avg_length:.1f}"
        )

    # Create comparative variance plot
    plot_comparative_variances(
        doc_variances, segment_variances, args.output, args.normalize
    )


if __name__ == "__main__":
    main()
