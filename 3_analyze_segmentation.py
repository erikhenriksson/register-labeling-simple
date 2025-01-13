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
    probabilities: np.ndarray, threshold: float = 0.5
) -> np.ndarray:
    """Convert probabilities to binary labels, marking all registers above threshold"""
    return (probabilities >= threshold).astype(np.float32)


def compute_register_variances(
    embeddings: np.ndarray, registers: np.ndarray, n_components: int = 10
) -> Dict[str, np.ndarray]:
    """Compute embedding variance for each register after PCA reduction"""
    n_registers = registers.shape[1]
    register_variances = []
    register_counts = []

    # Initialize PCA once for all data to maintain consistent components
    pca = PCA(n_components=n_components)
    # Fit PCA on all embeddings to get global principal components
    pca.fit(embeddings)

    # Transform all embeddings
    reduced_embeddings = pca.transform(embeddings)

    for reg_idx in range(n_registers):
        # Get embeddings where this register is present (prob >= threshold)
        mask = registers[:, reg_idx] == 1
        if np.sum(mask) > 0:
            reg_embeddings = reduced_embeddings[mask]
            # Compute variance across PCA components
            variance = np.mean(np.var(reg_embeddings, axis=0))
            count = np.sum(mask)

            # Optionally, we could weight the variances by explained variance ratio
            # variance = np.average(np.var(reg_embeddings, axis=0),
            #                      weights=pca.explained_variance_ratio_)
        else:
            variance = 0
            count = 0

        register_variances.append(variance)
        register_counts.append(count)

    # Also return explained variance ratio for analysis
    return {
        "variances": np.array(register_variances),
        "counts": np.array(register_counts),
        "explained_variance_ratio": pca.explained_variance_ratio_,
    }


def collect_data(
    input_jsonl_path: str, threshold: float = 0.5, limit: int = 100
) -> Dict[str, np.ndarray]:
    """Collect document and segment level data with multilabel register labeling"""
    document_embeddings = []
    document_registers = []
    segment_embeddings = []
    segment_registers = []

    with open(input_jsonl_path, "r") as f:
        for i, line in enumerate(f):
            if i >= limit:
                break

            data = json.loads(line)

            # Process document level
            doc_reg = convert_to_multilabel_registers(
                np.array(data["register_probabilities"]), threshold
            )
            document_embeddings.append(data["embedding"])
            document_registers.append(doc_reg)

            # Process segment level
            for emb, probs in zip(
                data["segmentation"]["embeddings"],
                data["segmentation"]["register_probabilities"],
            ):
                seg_reg = convert_to_multilabel_registers(np.array(probs), threshold)
                segment_embeddings.append(emb)
                segment_registers.append(seg_reg)

    return {
        "document_embeddings": np.array(document_embeddings),
        "document_registers": np.array(document_registers),
        "segment_embeddings": np.array(segment_embeddings),
        "segment_registers": np.array(segment_registers),
    }


def plot_results(doc_results: Dict, seg_results: Dict, output_path: str):
    """Plot comparison of embedding variances and PCA explained variance"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

    # Only plot registers that appear in both document and segment level
    mask = (doc_results["counts"] > 0) & (seg_results["counts"] > 0)
    register_indices = np.arange(len(doc_results["variances"]))[mask]
    doc_var = doc_results["variances"][mask]
    seg_var = seg_results["variances"][mask]
    register_names = [REGISTER_NAMES[i] for i in register_indices]

    # Plot 1: Variance comparison
    x = np.arange(len(register_names))
    ax1.bar(x - 0.2, doc_var, 0.4, label="Document Level")
    ax1.bar(x + 0.2, seg_var, 0.4, label="Segment Level")
    ax1.set_xlabel("Register")
    ax1.set_ylabel("Average PCA Component Variance")
    ax1.set_title("Embedding Variance by Register\n(After PCA)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(register_names, rotation=45, ha="right")
    ax1.legend()

    # Plot 2: Variance reduction
    variance_reduction = np.zeros_like(doc_var)
    nonzero_mask = doc_var > 0
    variance_reduction[nonzero_mask] = (
        (doc_var[nonzero_mask] - seg_var[nonzero_mask]) / doc_var[nonzero_mask] * 100
    )

    colors = ["green" if x > 0 else "red" for x in variance_reduction]
    ax2.bar(x, variance_reduction, color=colors)
    ax2.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax2.set_xlabel("Register")
    ax2.set_ylabel("Variance Reduction (%)")
    ax2.set_title("Reduction in Variance with Segmentation")
    ax2.set_xticks(x)
    ax2.set_xticklabels(register_names, rotation=45, ha="right")

    # Plot 3: PCA explained variance
    components = np.arange(1, len(doc_results["explained_variance_ratio"]) + 1)
    ax3.plot(
        components,
        np.cumsum(doc_results["explained_variance_ratio"]),
        label="Document Level",
        marker="o",
    )
    ax3.plot(
        components,
        np.cumsum(seg_results["explained_variance_ratio"]),
        label="Segment Level",
        marker="o",
    )
    ax3.set_xlabel("Number of Components")
    ax3.set_ylabel("Cumulative Explained Variance Ratio")
    ax3.set_title("PCA Explained Variance")
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main(input_path: str, output_path: str, threshold: float = 0.5, limit: int = 100):
    """Main analysis pipeline"""
    print("Loading and processing data...")
    data = collect_data(input_path, threshold, limit)

    print("\nComputing variances...")
    doc_results = compute_register_variances(
        data["document_embeddings"], data["document_registers"]
    )
    seg_results = compute_register_variances(
        data["segment_embeddings"], data["segment_registers"]
    )

    print("\nResults Summary:")
    print(
        f"{'Register':>8} {'Doc Count':>10} {'Seg Count':>10} {'Doc Var':>10} {'Seg Var':>10} {'% Reduction':>12}"
    )
    print("-" * 65)

    total_reduction = 0
    valid_registers = 0

    for reg_idx in range(len(doc_results["variances"])):
        doc_count = doc_results["counts"][reg_idx]
        seg_count = seg_results["counts"][reg_idx]

        if doc_count > 0 and seg_count > 0:
            doc_var = doc_results["variances"][reg_idx]
            seg_var = seg_results["variances"][reg_idx]
            # Add check for zero variance
            reduction = 0 if doc_var <= 0 else (doc_var - seg_var) / doc_var * 100
            reg_name = REGISTER_NAMES[reg_idx]
            print(
                f"{reg_name:>8} {doc_count:>10} {seg_count:>10} {doc_var:>10.3f} {seg_var:>10.3f} {reduction:>11.1f}%"
            )
            total_reduction += reduction
            valid_registers += 1

    # Add average reduction score
    print("-" * 65)
    avg_reduction = total_reduction / valid_registers if valid_registers > 0 else 0
    print(
        f"{'Average':>8} {'-':>10} {'-':>10} {'-':>10} {'-':>10} {avg_reduction:>11.1f}%"
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
