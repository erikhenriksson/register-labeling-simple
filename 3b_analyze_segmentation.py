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


def analyze_embeddings(data, level="document"):
    """Analyze embeddings at document or segment level"""
    register_embeddings = defaultdict(list)

    # First, let's check the embedding dimensions
    if level == "document":
        first_embedding = data[0]["embedding"]
        print(f"Embedding dimension: {len(first_embedding)}")
    else:
        first_embedding = data[0]["segmentation"]["embeddings"][0]
        print(f"Embedding dimension: {len(first_embedding)}")

    for item in data:
        if level == "document":
            probs = item["register_probabilities"]
            emb = item["embedding"]
            register = get_register_label(probs)
            if register:
                register_embeddings[register].append(emb)
        else:  # segment level
            for probs, emb in zip(
                item["segmentation"]["register_probabilities"],
                item["segmentation"]["embeddings"],
            ):
                register = get_register_label(probs)
                if register:
                    register_embeddings[register].append(emb)

    # Print number of examples per register before filtering
    print("\nNumber of examples per register before filtering:")
    for register, embeddings in register_embeddings.items():
        print(f"{register}: {len(embeddings)}")

    # Only keep registers with more than 1 example
    register_embeddings = {k: v for k, v in register_embeddings.items() if len(v) > 1}

    register_variances = {}
    for register, embeddings in register_embeddings.items():
        embeddings_array = np.array(embeddings)
        print(f"\nShape for register {register}: {embeddings_array.shape}")

        try:
            pca = PCA(n_components=30)
            pca_result = pca.fit_transform(embeddings_array)
            variances = np.var(pca_result, axis=0)
            register_variances[register] = np.mean(variances)
        except ValueError as e:
            print(f"Error processing register {register}: {e}")
            print(f"Data shape: {embeddings_array.shape}")

    return register_variances


def plot_variances(variances, title, output_path):
    plt.figure(figsize=(12, 6))
    registers = list(variances.keys())
    values = list(variances.values())

    plt.bar(registers, values)
    plt.title(f"Average Embedding Variance by Register ({title})")
    plt.xlabel("Register")
    plt.ylabel("Average Variance (first 30 PCA components)")
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the plot
    plt.savefig(f"{output_path}_{title.lower().replace(' ', '_')}.png")
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

    # Analyze at document level
    doc_variances = analyze_embeddings(data, level="document")
    print("\nDocument-level register variances:")
    for register, variance in sorted(doc_variances.items()):
        print(f"{register}: {variance:.4f}")
    plot_variances(doc_variances, "Document Level", args.output)

    # Analyze at segment level
    segment_variances = analyze_embeddings(data, level="segment")
    print("\nSegment-level register variances:")
    for register, variance in sorted(segment_variances.items()):
        print(f"{register}: {variance:.4f}")
    plot_variances(segment_variances, "Segment Level", args.output)


if __name__ == "__main__":
    main()
