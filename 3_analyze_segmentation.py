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
    """Analyze how much variance in embeddings is explained by registers"""
    pca = PCA(n_components=n_components)
    reduced_emb = pca.fit_transform(embeddings)

    r2_scores = []
    for pc in range(n_components):
        reg = LinearRegression()
        reg.fit(registers, reduced_emb[:, pc])
        r2_scores.append(reg.score(registers, reduced_emb[:, pc]))

    return {
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "register_r2_scores": np.array(r2_scores),
        "total_variance_explained": np.sum(pca.explained_variance_ratio_),
        "avg_register_r2": np.mean(r2_scores),
    }


def plot_results(doc_results: Dict, seg_results: Dict, output_path: str):
    """Plot comparison of document vs segment level results"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    x = np.arange(len(doc_results["explained_variance_ratio"]))

    # Plot explained variance
    ax1.plot(
        x,
        np.cumsum(doc_results["explained_variance_ratio"]),
        label="Document Level",
        marker="o",
    )
    ax1.plot(
        x,
        np.cumsum(seg_results["explained_variance_ratio"]),
        label="Segment Level",
        marker="o",
    )
    ax1.set_xlabel("Number of Components")
    ax1.set_ylabel("Cumulative Explained Variance Ratio")
    ax1.set_title("PCA Explained Variance")
    ax1.legend()

    # Plot R² scores
    ax2.plot(x, doc_results["register_r2_scores"], label="Document Level", marker="o")
    ax2.plot(x, seg_results["register_r2_scores"], label="Segment Level", marker="o")
    ax2.set_xlabel("Principal Component")
    ax2.set_ylabel("R² Score")
    ax2.set_title("Register Prediction of Principal Components")
    ax2.legend()

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
    print(f"Document Level:")
    print(
        f"- Total variance explained by {n_components} PCs: {doc_results['total_variance_explained']:.3f}"
    )
    print(f"- Average R² with registers: {doc_results['avg_register_r2']:.3f}")

    print(f"\nSegment Level:")
    print(
        f"- Total variance explained by {n_components} PCs: {seg_results['total_variance_explained']:.3f}"
    )
    print(f"- Average R² with registers: {seg_results['avg_register_r2']:.3f}")

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
