import json
import numpy as np
from typing import Dict
import matplotlib.pyplot as plt

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


def convert_to_dominant_registers(
    probabilities: np.ndarray, threshold: float = 0.5
) -> np.ndarray:
    """Convert probabilities to binary labels, keeping only dominant registers above threshold"""
    binary = np.zeros_like(probabilities)
    max_prob = np.max(probabilities)
    if max_prob >= threshold:
        binary[probabilities >= max_prob] = 1
    return binary


def collect_data(
    input_jsonl_path: str, threshold: float = 0.5, limit: int = 100
) -> Dict[str, np.ndarray]:
    """Collect document and segment level data with dominant register labeling"""
    document_embeddings = []
    document_registers = []
    segment_embeddings = []
    segment_registers = []

    # Add counters for debugging
    total_segments = 0
    segments_per_doc = []

    with open(input_jsonl_path, "r") as f:
        for i, line in enumerate(f):
            if i >= limit:
                break

            data = json.loads(line)

            # Process document level
            doc_reg = convert_to_dominant_registers(
                np.array(data["register_probabilities"]), threshold
            )
            document_embeddings.append(data["embedding"])
            document_registers.append(doc_reg)

            # Count segments in this document
            num_segments = len(data["segmentation"]["embeddings"])
            segments_per_doc.append(num_segments)
            total_segments += num_segments

            # Process segment level
            for emb, probs in zip(
                data["segmentation"]["embeddings"],
                data["segmentation"]["register_probabilities"],
            ):
                seg_reg = convert_to_dominant_registers(np.array(probs), threshold)
                segment_embeddings.append(emb)
                segment_registers.append(seg_reg)

    # Print diagnostic information
    print(f"\nDiagnostic Information:")
    print(f"Total documents processed: {len(document_embeddings)}")
    print(f"Total segments found: {total_segments}")
    print(f"Average segments per document: {np.mean(segments_per_doc):.2f}")
    print(
        f"Min/Max segments per document: {min(segments_per_doc)}/{max(segments_per_doc)}"
    )

    # Print register distribution
    doc_reg_dist = np.sum(np.array(document_registers), axis=0)
    seg_reg_dist = np.sum(np.array(segment_registers), axis=0)
    print("\nRegister distribution (doc level vs segment level):")
    for i, reg in enumerate(REGISTER_NAMES):
        print(f"{reg:>4}: {doc_reg_dist[i]:>4} docs, {seg_reg_dist[i]:>4} segments")

    return {
        "document_embeddings": np.array(document_embeddings),
        "document_registers": np.array(document_registers),
        "segment_embeddings": np.array(segment_embeddings),
        "segment_registers": np.array(segment_registers),
    }


def compute_register_variances(
    embeddings: np.ndarray, registers: np.ndarray
) -> Dict[str, np.ndarray]:
    """Compute embedding variance for each dominant register"""
    n_registers = registers.shape[1]
    register_variances = []
    register_counts = []

    for reg_idx in range(n_registers):
        # Get embeddings where this register is dominant
        mask = registers[:, reg_idx] == 1
        if np.sum(mask) > 0:  # If we have any instances of this register
            reg_embeddings = embeddings[mask]
            # Compute variance across all embedding dimensions
            variance = np.mean(np.var(reg_embeddings, axis=0))
            count = np.sum(mask)
        else:
            variance = 0
            count = 0

        register_variances.append(variance)
        register_counts.append(count)

    return {
        "variances": np.array(register_variances),
        "counts": np.array(register_counts),
    }


def plot_results(doc_results: Dict, seg_results: Dict, output_path: str):
    """Plot comparison of embedding variances for each dominant register"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

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
    ax1.set_ylabel("Average Embedding Variance")
    ax1.set_title("Embedding Variance by Register")
    ax1.set_xticks(x)
    ax1.set_xticklabels(register_names, rotation=45, ha="right")
    ax1.legend()

    # Plot 2: Variance reduction
    # Add check for zero variance
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
