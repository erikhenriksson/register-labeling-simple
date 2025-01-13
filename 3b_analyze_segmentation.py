import json
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import List, Dict, Tuple

# The label hierarchy
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

# Flat list of labels
labels_all = [k for k in labels_structure.keys()] + [
    item for row in labels_structure.values() for item in row
]

def get_parent_indices(label_structure: Dict) -> Dict[int, int]:
    """Create mapping of child indices to parent indices"""
    parent_map = {}
    for parent, children in label_structure.items():
        parent_idx = labels_all.index(parent)
        for child in children:
            child_idx = labels_all.index(child)
            parent_map[child_idx] = parent_idx
    return parent_map

def process_probabilities(probs: List[float], threshold: float = 0.4) -> np.ndarray:
    """
    Convert probabilities to labels considering hierarchy.
    Returns array of 0s and 1s indicating label presence.
    """
    parent_indices = get_parent_indices(labels_structure)
    labels = (np.array(probs) > threshold).astype(int)
    
    # Zero out parent when child is active
    for child_idx, parent_idx in parent_indices.items():
        if labels[child_idx] == 1:
            labels[parent_idx] = 0
            
    return labels

def read_jsonl(file_path: str) -> List[dict]:
    """Read JSONL file and return list of records"""
    encodings = ['utf-8', 'latin1', 'cp1252']
    data = []
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                for line in f:
                    try:
                        data.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        continue
            return data  # If successful, return the data
        except UnicodeDecodeError:
            continue  # Try next encoding if current one fails
    
    raise ValueError(f"Could not read file {file_path} with any of the attempted encodings: {encodings}")

def get_label_from_probabilities(probs: List[float]) -> Tuple[str, int]:
    """
    Get single label from probability array.
    Returns tuple of (label_name, label_index).
    """
    labels = process_probabilities(probs)
    label_indices = np.where(labels == 1)[0]
    
    # Only return if exactly one label
    if len(label_indices) == 1:
        label_idx = label_indices[0]
        return labels_all[label_idx], label_idx
    return None, None

def collect_register_data(data: List[dict], is_segment: bool = False) -> Dict:
    """
    Collect embeddings by register from data.
    Returns dict mapping registers to lists of embeddings.
    """
    register_data = defaultdict(list)
    
    for item in data:
        if is_segment:
            # Handle segmented data
            for seg_probs, seg_emb in zip(item['segmentation']['register_probabilities'], 
                                        item['segmentation']['embeddings']):
                label, _ = get_label_from_probabilities(seg_probs)
                if label:
                    register_data[label].append(seg_emb)
        else:
            # Handle document-level data
            label, _ = get_label_from_probabilities(item['register_probabilities'])
            if label:
                register_data[label].append(item['embedding'])
    
    return register_data

def calculate_variances(register_data: Dict, n_components: int = 50) -> Dict:
    """
    Calculate variances of first n PCA components for each register.
    Returns dict mapping registers to their variance values.
    """
    variances = {}
    
    for register, embeddings in register_data.items():
        if len(embeddings) > 1:  # Only process registers with multiple examples
            embeddings_array = np.array(embeddings)
            pca = PCA(n_components=min(n_components, len(embeddings_array)))
            transformed = pca.fit_transform(embeddings_array)
            variances[register] = np.var(transformed, axis=0)
            
    return variances

def plot_register_variances(doc_variances: Dict, seg_variances: Dict, output_path: str = 'variances.png'):
    """Plot variance comparison between document and segment level"""
    registers = sorted(set(doc_variances.keys()) | set(seg_variances.keys()))
    
    # Prepare data for plotting
    doc_means = [np.mean(doc_variances.get(reg, [0])) for reg in registers]
    seg_means = [np.mean(seg_variances.get(reg, [0])) for reg in registers]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(registers))
    width = 0.35
    
    ax.bar(x - width/2, doc_means, width, label='Document-level')
    ax.bar(x + width/2, seg_means, width, label='Segment-level')
    
    ax.set_ylabel('Mean Variance')
    ax.set_title('Register Embedding Variances: Document vs Segment Level')
    ax.set_xticks(x)
    ax.set_xticklabels(registers, rotation=45)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def print_variance_table(doc_variances: Dict, seg_variances: Dict):
    """Print formatted table of variances"""
    registers = sorted(set(doc_variances.keys()) | set(seg_variances.keys()))
    
    print("\nRegister Variance Comparison:")
    print(f"{'Register':<10} {'Doc Mean':<15} {'Seg Mean':<15} {'Doc Count':<10} {'Seg Count':<10}")
    print("-" * 60)
    
    for reg in registers:
        doc_mean = np.mean(doc_variances.get(reg, [0])) if reg in doc_variances else 0
        seg_mean = np.mean(seg_variances.get(reg, [0])) if reg in seg_variances else 0
        doc_count = len(doc_variances.get(reg, [])) if reg in doc_variances else 0
        seg_count = len(seg_variances.get(reg, [])) if reg in seg_variances else 0
        
        print(f"{reg:<10} {doc_mean:<15.4f} {seg_mean:<15.4f} {doc_count:<10} {seg_count:<10}")

def main(doc_file: str, seg_file: str):
    """Main function to process files and generate analysis"""
    # Read data
    doc_data = read_jsonl(doc_file)
    seg_data = read_jsonl(seg_file)
    
    # Collect embeddings by register
    doc_register_data = collect_register_data(doc_data, is_segment=False)
    seg_register_data = collect_register_data(seg_data, is_segment=True)
    
    # Calculate variances
    doc_variances = calculate_variances(doc_register_data)
    seg_variances = calculate_variances(seg_register_data)
    
    # Generate outputs
    print_variance_table(doc_variances, seg_variances)
    plot_register_variances(doc_variances, seg_variances)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python script.py <doc_file.jsonl> <seg_file.jsonl>")
        sys.exit(1)
    
    main(sys.argv[1], sys.argv[2])