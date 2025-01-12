import json
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from typing import Dict
import matplotlib.pyplot as plt


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

            # Process segment level
            for emb, probs in zip(
                data["segmentation"]["embeddings"],
                data["segmentation"]["register_probabilities"],
            ):
                seg_reg = convert_to_dominant_registers(np.array(probs), threshold)
                segment_embeddings.append(emb)
                segment_registers.append(seg_reg)

    print(
        f"Processed {len(document_embeddings)} documents and {len(segment_embeddings)} segments"
    )
    return {
        "document_embeddings": np.array(document_embeddings),
        "document_registers": np.array(document_registers),
        "segment_embeddings": np.array(segment_embeddings),
        "segment_registers": np.array(segment_registers),
    }


def analyze_register_variance(
    embeddings: np.ndarray, registers: np.ndarray, n_components: int = 10
) -> Dict[str, np.ndarray]:
    """Analyze how each register individually explains variance in the embeddings"""
    pca = PCA(n_components=n_components)
    reduced_emb = pca.fit_transform(embeddings)

    # Overall PCA stats
    results = {
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "total_variance_explained": np.sum(pca.explained_variance_ratio_),
    }

    # Analyze each register individually
    n_registers = registers.shape[1]
    register_r2_by_component = np.zeros((n_registers, n_components))

    for reg_idx in range(n_registers):
        for pc in range(n_components):
            reg = LinearRegression()
            reg.fit(registers[:, reg_idx : reg_idx + 1], reduced_emb[:, pc])
            register_r2_by_component[reg_idx, pc] = reg.score(
                registers[:, reg_idx : reg_idx + 1], reduced_emb[:, pc]
            )

    results["register_r2_by_component"] = register_r2_by_component
    results["avg_r2_by_register"] = np.mean(register_r2_by_component, axis=1)

    return results


def plot_results(doc_results: Dict, seg_results: Dict, output_path: str):
    """Plot comparison of how each dominant register explains variance"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Average R² by register (bar plot comparison)
    n_registers = len(doc_results["avg_r2_by_register"])
    register_x = np.arange(n_registers)
    ax1.bar(
        register_x - 0.2, doc_results["avg_r2_by_register"], 0.4, label="Document Level"
    )
    ax1.bar(
        register_x + 0.2, seg_results["avg_r2_by_register"], 0.4, label="Segment Level"
    )
    ax1.set_xlabel("Register Index")
    ax1.set_ylabel("Average R² across PCs")
    ax1.set_title("Register Explanatory Power")
    ax1.legend()

    # Plot 2: R² difference (segment - document) to show improvement
    r2_diff = seg_results["avg_r2_by_register"] - doc_results["avg_r2_by_register"]
    colors = ["green" if x > 0 else "red" for x in r2_diff]
    ax2.bar(register_x, r2_diff, color=colors)
    ax2.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax2.set_xlabel("Register Index")
    ax2.set_ylabel("R² Difference (Segment - Document)")
    ax2.set_title("Improvement in Register Explanatory Power\nwith Segmentation")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main(
    input_path: str,
    output_path: str,
    threshold: float = 0.5,
    n_components: int = 10,
    limit: int = 100,
):
    """Main analysis pipeline"""
    print("Loading and processing data...")
    data = collect_data(input_path, threshold, limit)

    print("\nAnalyzing document level data...")
    doc_results = analyze_register_variance(
        data["document_embeddings"], data["document_registers"], n_components
    )

    print("\nAnalyzing segment level data...")
    seg_results = analyze_register_variance(
        data["segment_embeddings"], data["segment_registers"], n_components
    )

    # Print summary statistics
    print("\nResults Summary:")
    print(f"\nRegister-wise R² values (averaged across PCs):")
    print(f"{'Register':>8} {'Document':>10} {'Segment':>10} {'Difference':>12}")
    print("-" * 42)
    for reg_idx in range(len(doc_results["avg_r2_by_register"])):
        doc_r2 = doc_results["avg_r2_by_register"][reg_idx]
        seg_r2 = seg_results["avg_r2_by_register"][reg_idx]
        diff = seg_r2 - doc_r2
        print(f"{reg_idx:>8} {doc_r2:>10.3f} {seg_r2:>10.3f} {diff:>12.3f}")

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
        "--n-components",
        type=int,
        default=10,
        help="Number of principal components to analyze",
    )
    parser.add_argument(
        "--limit", type=int, default=100, help="Maximum number of documents to process"
    )
    args = parser.parse_args()

    main(
        args.input_file, args.output_file, args.threshold, args.n_components, args.limit
    )
