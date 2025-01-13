import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Label structure
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


def get_single_labels(probs, threshold=0.4):
    """Convert probabilities to labels considering hierarchy."""
    labels = (np.array(probs) > threshold).astype(int)

    # Create parent-child mapping
    parent_child_map = {}
    current_idx = len(labels_structure)
    for parent, children in labels_structure.items():
        parent_idx = list(labels_structure.keys()).index(parent)
        for _ in children:
            parent_child_map[current_idx] = parent_idx
            current_idx += 1

    # Zero out parent when child is active
    for child_idx, parent_idx in parent_child_map.items():
        if labels[child_idx] == 1:
            labels[parent_idx] = 0

    return labels


def process_embeddings(embeddings, labels):
    """Process embeddings using PCA and calculate variances."""
    # Perform PCA
    pca = PCA(n_components=50)
    embeddings_pca = pca.fit_transform(embeddings)

    # Calculate variances per register
    register_variances = {}
    for i, label in enumerate(labels_all):
        # Get embeddings for this register
        mask = np.array([l[i] for l in labels]) == 1
        if np.sum(mask) > 1:  # Only calculate if we have more than 1 example
            register_embeddings = embeddings_pca[mask]
            variances = np.var(register_embeddings, axis=0)
            register_variances[label] = np.mean(variances)

    return register_variances


def visualize_variances(variances):
    """Create bar plot of variances."""
    plt.figure(figsize=(12, 6))
    labels = list(variances.keys())
    values = list(variances.values())

    plt.bar(labels, values)
    plt.xticks(rotation=45, ha="right")
    plt.title("Average Embedding Variance by Register")
    plt.ylabel("Average Variance (First 50 PCA Components)")
    plt.tight_layout()

    # Print table
    print("\nRegister Variances:")
    print("-" * 40)
    print(f"{'Register':<15} {'Variance':>10}")
    print("-" * 40)
    for label, value in sorted(variances.items()):
        print(f"{label:<15} {value:>10.4f}")


def analyze_document(text, register_probs, embedding):
    """Analyze a single document."""
    # Convert probabilities to labels
    labels = get_single_labels(register_probs)

    # Get active registers
    active_registers = [labels_all[i] for i, l in enumerate(labels) if l == 1]
    return labels, active_registers


# Process first document
print("\nProcessing document:")
labels, registers = analyze_document(
    data["text"], data["register_probabilities"], data["embedding"]
)

print("\nActive registers:", registers)

# Process segments
segment_labels = []
segment_embeddings = []
for i, (probs, emb) in enumerate(
    zip(
        data["segmentation"]["register_probabilities"],
        data["segmentation"]["embeddings"],
    )
):
    labels, registers = analyze_document(data["segmentation"]["texts"][i], probs, emb)
    segment_labels.append(labels)
    segment_embeddings.append(emb)

# Calculate variances for document and segments
print("\nCalculating variances...")
segment_embeddings = np.array(segment_embeddings)
segment_labels = np.array(segment_labels)

document_variances = process_embeddings(
    np.array([data["embedding"]]), np.array([labels])
)
segment_variances = process_embeddings(segment_embeddings, segment_labels)

print("\nDocument-level variances:")
for register, variance in document_variances.items():
    print(f"{register}: {variance:.4f}")

print("\nSegment-level variances:")
visualize_variances(segment_variances)
