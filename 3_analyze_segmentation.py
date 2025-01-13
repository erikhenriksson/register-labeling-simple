import json
import numpy as np
from typing import Dict
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Register names mapping
REGISTER_NAMES = [
    "MT",
    "LY",
    "SP",
    "ID",
    "NA",
    "HI",
    "IN",
    "OP",
    "IP",  # Main categories
    "it",  # SP (Spoken)
    "ne",
    "sr",
    "nb",  # NA (Narrative)
    "re",  # HI (How-to/Instructional)
    "en",
    "ra",
    "dtp",
    "fi",
    "lt",  # IN (Informational)
    "rv",
    "ob",
    "rs",
    "av",  # OP (Opinion)
    "ds",
    "ed",  # IP (Interactive/Interpersonal)
]


def convert_to_multilabel_registers(
    probabilities: np.ndarray, threshold: float = 0.5, exclude_hybrids: bool = True
) -> np.ndarray:
    """
    Convert probabilities to binary labels, handling hierarchical structure.
    When a child register is active, its parent is set to 0.
    If exclude_hybrids is True, only rows with exactly one 1 are kept.
    """
    # Handle different input shapes
    if len(probabilities.shape) == 3:
        probabilities = probabilities.reshape(probabilities.shape[0], -1)
    elif len(probabilities.shape) == 1:
        probabilities = probabilities.reshape(1, -1)

    # First convert to binary based on threshold
    labels = (probabilities >= threshold).astype(np.float32)

    # Define parent-child relationships
    hierarchy = {
        "SP": [REGISTER_NAMES.index("it")],
        "NA": [REGISTER_NAMES.index(x) for x in ["ne", "sr", "nb"]],
        "HI": [REGISTER_NAMES.index("re")],
        "IN": [REGISTER_NAMES.index(x) for x in ["en", "ra", "dtp", "fi", "lt"]],
        "OP": [REGISTER_NAMES.index(x) for x in ["rv", "ob", "rs", "av"]],
        "IP": [REGISTER_NAMES.index(x) for x in ["ds", "ed"]],
    }

    # For each parent, if any child is 1, set parent to 0
    for parent, children in hierarchy.items():
        parent_idx = REGISTER_NAMES.index(parent)
        for row_idx in range(labels.shape[0]):
            if any(labels[row_idx, child_idx] == 1 for child_idx in children):
                labels[row_idx, parent_idx] = 0

    if exclude_hybrids:
        # Keep only rows with exactly one 1
        row_sums = np.sum(labels, axis=1)
        hybrid_mask = row_sums == 1
        labels = labels * hybrid_mask[:, np.newaxis]  # Zero out hybrid rows

    return labels  # Always return 2D array


def compute_length_normalized_variances(
    embeddings: np.ndarray,
    registers: np.ndarray,
    lengths: np.ndarray,
    n_components: int = 20,
) -> Dict[str, np.ndarray]:
    """
    Compute length-normalized embedding variance for each register after PCA reduction
    """
    print(f"Debug - Input shapes:")
    print(f"embeddings shape: {embeddings.shape}")
    print(f"registers shape: {registers.shape}")
    print(f"lengths shape: {lengths.shape}")

    # Ensure registers is 2D
    if len(registers.shape) == 1:
        registers = registers.reshape(1, -1)
        embeddings = embeddings.reshape(1, -1)
        lengths = np.array([lengths])
    if len(registers.shape) == 3:
        registers = registers.reshape(registers.shape[0], registers.shape[2])

    n_registers = registers.shape[1]
    register_variances = []
    register_counts = []
    register_raw_variances = []
    register_avg_lengths = []

    # Initialize and fit PCA
    pca = PCA(n_components=n_components)
    pca.fit(embeddings)
    reduced_embeddings = pca.transform(embeddings)
    print(f"reduced_embeddings shape: {reduced_embeddings.shape}")

    for reg_idx in range(n_registers):
        mask = registers[:, reg_idx] == 1
        print(f"\nRegister {REGISTER_NAMES[reg_idx]}:")
        print(f"mask shape: {mask.shape}")
        print(f"Number of True in mask: {np.sum(mask)}")

        if np.sum(mask) > 0:
            # Use boolean indexing for first dimension only
            reg_embeddings = reduced_embeddings[mask]
            reg_lengths = lengths[mask]

            print(f"reg_embeddings shape: {reg_embeddings.shape}")
            print(f"reg_lengths shape: {reg_lengths.shape}")

            # Compute raw variance
            raw_variance = np.mean(np.var(reg_embeddings, axis=0))

            # Compute length-normalized variance
            avg_length = np.mean(reg_lengths)
            normalized_variance = raw_variance * np.sqrt(avg_length)

            count = np.sum(mask)
        else:
            raw_variance = 0
            normalized_variance = 0
            count = 0
            avg_length = 0

        register_variances.append(normalized_variance)
        register_raw_variances.append(raw_variance)
        register_counts.append(count)
        register_avg_lengths.append(avg_length)

    return {
        "normalized_variances": np.array(register_variances),
        "raw_variances": np.array(register_raw_variances),
        "counts": np.array(register_counts),
        "avg_lengths": np.array(register_avg_lengths),
        "explained_variance_ratio": pca.explained_variance_ratio_,
    }

    return {
        "normalized_variances": np.array(register_variances),
        "raw_variances": np.array(register_raw_variances),
        "counts": np.array(register_counts),
        "avg_lengths": np.array(register_avg_lengths),
        "explained_variance_ratio": pca.explained_variance_ratio_,
    }


def plot_results(doc_results: Dict, seg_results: Dict, output_path: str):
    """Plot comparison of raw and normalized variances"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))

    # Only plot registers that appear in both document and segment level
    mask = (doc_results["counts"] > 0) & (seg_results["counts"] > 0)
    register_indices = np.arange(len(doc_results["normalized_variances"]))[mask]
    doc_var = doc_results["raw_variances"][mask]
    seg_var = seg_results["raw_variances"][mask]
    doc_norm_var = doc_results["normalized_variances"][mask]
    seg_norm_var = seg_results["normalized_variances"][mask]
    register_names = [REGISTER_NAMES[i] for i in register_indices]

    # Plot 1: Raw variance comparison
    x = np.arange(len(register_names))
    ax1.bar(x - 0.2, doc_var, 0.4, label="Document Level")
    ax1.bar(x + 0.2, seg_var, 0.4, label="Segment Level")
    ax1.set_xlabel("Register")
    ax1.set_ylabel("Raw Variance")
    ax1.set_title("Raw Embedding Variance by Register\n(After PCA)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(register_names, rotation=45, ha="right")
    ax1.legend()

    # Plot 2: Normalized variance comparison
    ax2.bar(x - 0.2, doc_norm_var, 0.4, label="Document Level")
    ax2.bar(x + 0.2, seg_norm_var, 0.4, label="Segment Level")
    ax2.set_xlabel("Register")
    ax2.set_ylabel("Length-Normalized Variance")
    ax2.set_title("Length-Normalized Embedding Variance by Register")
    ax2.set_xticks(x)
    ax2.set_xticklabels(register_names, rotation=45, ha="right")
    ax2.legend()

    # Plot 3: Length vs Variance scatter
    ax3.scatter(doc_results["avg_lengths"][mask], doc_var, label="Documents", alpha=0.6)
    ax3.scatter(seg_results["avg_lengths"][mask], seg_var, label="Segments", alpha=0.6)
    ax3.set_xlabel("Average Text Length")
    ax3.set_ylabel("Raw Variance")
    ax3.set_title("Text Length vs Raw Variance")
    ax3.legend()

    # Plot 4: PCA explained variance
    components = np.arange(1, len(doc_results["explained_variance_ratio"]) + 1)
    ax4.plot(
        components,
        np.cumsum(doc_results["explained_variance_ratio"]),
        label="Document Level",
        marker="o",
    )
    ax4.plot(
        components,
        np.cumsum(seg_results["explained_variance_ratio"]),
        label="Segment Level",
        marker="o",
    )
    ax4.set_xlabel("Number of Components")
    ax4.set_ylabel("Cumulative Explained Variance Ratio")
    ax4.set_title("PCA Explained Variance")
    ax4.legend()
    ax4.grid(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def collect_data(
    input_jsonl_path: str, threshold: float = 0.5, limit: int = 100
) -> Dict[str, np.ndarray]:
    """Collect document and segment level data including text lengths"""
    document_embeddings = []
    document_registers = []
    document_lengths = []
    segment_embeddings = []
    segment_registers = []
    segment_lengths = []

    total_docs = 0
    pure_docs = 0
    total_segs = 0
    pure_segs = 0

    with open(input_jsonl_path, "r") as f:
        for i, line in enumerate(f):
            if i >= limit:
                break

            data = json.loads(line)

            # Process document level
            doc_reg = convert_to_multilabel_registers(
                np.array(data["register_probabilities"]), threshold
            )
            if np.sum(doc_reg) > 0:  # If not zeroed out as hybrid
                document_embeddings.append(data["embedding"])
                document_registers.append(doc_reg)
                document_lengths.append(len(data["text"].split()))
                pure_docs += 1
            total_docs += 1

            # Process segment level
            for emb, probs, text in zip(
                data["segmentation"]["embeddings"],
                data["segmentation"]["register_probabilities"],
                data["segmentation"]["texts"],
            ):
                seg_reg = convert_to_multilabel_registers(np.array(probs), threshold)
                if np.sum(seg_reg) > 0:  # If not zeroed out as hybrid
                    segment_embeddings.append(emb)
                    segment_registers.append(seg_reg)
                    segment_lengths.append(len(text.split()))
                    pure_segs += 1
                total_segs += 1

    print(f"\nPurity Statistics:")
    print(f"Documents: {pure_docs}/{total_docs} pure ({pure_docs/total_docs*100:.1f}%)")
    print(f"Segments: {pure_segs}/{total_segs} pure ({pure_segs/total_segs*100:.1f}%)")

    return {
        "document_embeddings": np.array(document_embeddings),
        "document_registers": np.array(document_registers),
        "document_lengths": np.array(document_lengths),
        "segment_embeddings": np.array(segment_embeddings),
        "segment_registers": np.array(segment_registers),
        "segment_lengths": np.array(segment_lengths),
    }


def main(input_path: str, output_path: str, threshold: float = 0.5, limit: int = 100):
    """Main analysis pipeline"""
    print("Loading and processing data...")
    data = collect_data(input_path, threshold, limit)

    print("\nComputing variances...")
    doc_results = compute_length_normalized_variances(
        data["document_embeddings"],
        data["document_registers"],
        data["document_lengths"],
    )
    seg_results = compute_length_normalized_variances(
        data["segment_embeddings"], data["segment_registers"], data["segment_lengths"]
    )

    print("\nResults Summary:")
    print(
        f"{'Register':>8} {'Doc Count':>10} {'Seg Count':>10} {'Raw Doc':>10} {'Raw Seg':>10} {'Raw Red%':>10} {'Norm Doc':>10} {'Norm Seg':>10} {'Norm Red%':>10}"
    )
    print("-" * 105)

    total_raw_reduction = 0
    total_norm_reduction = 0
    valid_registers = 0

    for reg_idx in range(len(doc_results["normalized_variances"])):
        doc_count = doc_results["counts"][reg_idx]
        seg_count = seg_results["counts"][reg_idx]

        if doc_count > 0 and seg_count > 0:
            raw_doc = doc_results["raw_variances"][reg_idx]
            raw_seg = seg_results["raw_variances"][reg_idx]
            norm_doc = doc_results["normalized_variances"][reg_idx]
            norm_seg = seg_results["normalized_variances"][reg_idx]

            # Only compute reductions if we have valid variances
            if not (np.isnan(raw_doc) or np.isnan(raw_seg)):
                raw_reduction = (
                    ((raw_doc - raw_seg) / raw_doc * 100) if raw_doc > 0 else np.nan
                )
                norm_reduction = (
                    ((norm_doc - norm_seg) / norm_doc * 100) if norm_doc > 0 else np.nan
                )

                if not np.isnan(raw_reduction) and not np.isnan(norm_reduction):
                    total_raw_reduction += raw_reduction
                    total_norm_reduction += norm_reduction
                    valid_registers += 1
            else:
                raw_reduction = np.nan
                norm_reduction = np.nan

            reg_name = REGISTER_NAMES[reg_idx]
            # Format values, showing 'n/a' for NaN
            raw_doc_str = f"{raw_doc:>10.3f}" if not np.isnan(raw_doc) else "      n/a"
            raw_seg_str = f"{raw_seg:>10.3f}" if not np.isnan(raw_seg) else "      n/a"
            raw_red_str = (
                f"{raw_reduction:>9.1f}%"
                if not np.isnan(raw_reduction)
                else "      n/a"
            )
            norm_doc_str = (
                f"{norm_doc:>10.3f}" if not np.isnan(norm_doc) else "      n/a"
            )
            norm_seg_str = (
                f"{norm_seg:>10.3f}" if not np.isnan(norm_seg) else "      n/a"
            )
            norm_red_str = (
                f"{norm_reduction:>9.1f}%"
                if not np.isnan(norm_reduction)
                else "      n/a"
            )

            print(
                f"{reg_name:>8} {doc_count:>10} {seg_count:>10} "
                f"{raw_doc_str} {raw_seg_str} {raw_red_str} "
                f"{norm_doc_str} {norm_seg_str} {norm_red_str}"
            )

    # Add average reduction scores (only for valid registers)
    print("-" * 105)
    avg_raw_reduction = (
        total_raw_reduction / valid_registers if valid_registers > 0 else np.nan
    )
    avg_norm_reduction = (
        total_norm_reduction / valid_registers if valid_registers > 0 else np.nan
    )
    avg_raw_str = (
        f"{avg_raw_reduction:>9.1f}%"
        if not np.isnan(avg_raw_reduction)
        else "      n/a"
    )
    avg_norm_str = (
        f"{avg_norm_reduction:>9.1f}%"
        if not np.isnan(avg_norm_reduction)
        else "      n/a"
    )
    print(
        f"{'Average':>8} {'-':>10} {'-':>10} "
        f"{'-':>10} {'-':>10} {avg_raw_str} "
        f"{'-':>10} {'-':>10} {avg_norm_str}"
    )

    # Create and save plot
    plot_results(doc_results, seg_results, output_path)
    print(f"\nPlot saved to: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="Path to segmentation output JSONL")
    parser.add_argument("output_file", help="Path to save the plot (PNG)")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold for dominant register selection",
    )
    parser.add_argument(
        "--limit", type=int, default=100, help="Maximum number of documents to process"
    )
    args = parser.parse_args()

    main(args.input_file, args.output_file, args.threshold, args.limit)
