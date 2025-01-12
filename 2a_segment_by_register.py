import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set
import numpy as np
import spacy
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


@dataclass
class Segment:
    text: str
    start_idx: int
    end_idx: int
    register_probs: List[float]
    embedding: Optional[List[float]] = None


class RegisterSegmenter:
    def __init__(
        self,
        min_segment_length: int = 300,
        register_threshold: float = 0.4,
        batch_size: int = 32,
    ):
        # Load spaCy for sentence splitting
        self.nlp = spacy.load("xx_ent_wiki_sm")
        if "sentencizer" not in self.nlp.pipe_names:
            self.nlp.add_pipe("sentencizer")

        self.min_segment_length = min_segment_length
        self.register_threshold = register_threshold
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model and tokenizer
        model_path = "/scratch/project_2011770/bge-2048"
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")
        self.model = self.model.to(self.device)
        self.model.eval()
        self.model.config.output_hidden_states = True
        self.id2label = self.model.config.id2label

        # Set up hierarchical label structure
        self.labels_structure = {
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

        # Create reverse mapping: child -> parent
        self.child_to_parent = {}
        for parent, children in self.labels_structure.items():
            for child in children:
                self.child_to_parent[child] = parent

        # Create label to index mapping
        self.label_to_idx = {label: idx for idx, label in self.id2label.items()}

    def predict_batch(
        self, texts: List[str]
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Predict register probabilities and embeddings for a batch of texts"""
        with torch.no_grad():
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=2048,
                return_tensors="pt",
            ).to(self.device)
            outputs = self.model(**inputs)

            # Get probabilities
            probs = torch.sigmoid(outputs.logits).cpu().numpy()
            # Get embeddings
            embeddings = outputs.hidden_states[-1][:, 0, :].cpu().numpy()

            # Adjust probabilities for hierarchy
            adjusted_probs = [self.adjust_probs_for_hierarchy(p) for p in probs]

            return adjusted_probs, embeddings

    def predict_registers_and_embeddings(
        self, text: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict both register probabilities and embeddings for a single text segment"""
        probs, embeddings = self.predict_batch([text])
        return probs[0], embeddings[0]

    def compute_entropy(self, probs: np.ndarray) -> float:
        """
        Compute entropy for multilabel probabilities.
        For each label position, we have a binary probability (p, 1-p).
        Total entropy is the sum of entropies across all label positions.
        """
        eps = 1e-10  # Small constant to avoid log(0)
        probs = np.clip(probs, eps, 1.0 - eps)
        binary_entropies = -(probs * np.log(probs) + (1 - probs) * np.log(1 - probs))
        return float(np.sum(binary_entropies))

    def adjust_probs_for_hierarchy(self, probs: np.ndarray) -> np.ndarray:
        """
        Adjust probabilities to enforce hierarchical consistency:
        - If any child label is above threshold, set its parent to 0
        """
        adjusted_probs = probs.copy()

        for parent, children in self.labels_structure.items():
            if not children:
                continue

            parent_idx = self.label_to_idx[parent]
            children_indices = [self.label_to_idx[child] for child in children]

            if any(
                adjusted_probs[idx] >= self.register_threshold
                for idx in children_indices
            ):
                adjusted_probs[parent_idx] = 0.0

        return adjusted_probs

    def get_dominant_registers(self, probs: np.ndarray) -> Set[str]:
        """Get set of dominant registers considering hierarchical relationships"""
        dominant = set()
        for i, prob in enumerate(probs):
            if prob >= self.register_threshold:
                label = self.id2label[i]
                if label in self.child_to_parent:
                    dominant.add(label)
                else:
                    children = self.labels_structure.get(label, [])
                    child_indices = [self.label_to_idx[child] for child in children]
                    if not any(
                        probs[idx] >= self.register_threshold for idx in child_indices
                    ):
                        dominant.add(label)
        return dominant

    def split_to_sentences(self, text: str) -> List[str]:
        """Split text into sentences using spaCy"""
        doc = self.nlp(text)
        return [sent.text.strip() for sent in doc.sents]

    def get_valid_segmentations(
        self,
        sentences: List[str],
    ) -> List[Tuple[Segment, Segment]]:
        """Get all valid segmentations based on minimum length constraint with batched predictions"""
        valid_splits = []
        total_sentences = len(sentences)

        # First collect all potential segments
        segments_to_predict = []
        split_indices = []

        for split_idx in range(1, total_sentences):
            segment1_text = " ".join(sentences[:split_idx])
            segment2_text = " ".join(sentences[split_idx:])

            if (
                len(segment1_text) >= self.min_segment_length
                and len(segment2_text) >= self.min_segment_length
            ):
                segments_to_predict.extend([segment1_text, segment2_text])
                split_indices.append(split_idx)

        if not segments_to_predict:
            return []

        # Predict in batches
        all_probs = []
        all_embeddings = []

        for i in range(0, len(segments_to_predict), self.batch_size):
            batch_texts = segments_to_predict[i : i + self.batch_size]
            probs, embeddings = self.predict_batch(batch_texts)
            all_probs.extend(probs)
            all_embeddings.extend(embeddings)

        # Create segments
        valid_segmentations = []
        for idx, split_idx in enumerate(split_indices):
            seg1 = Segment(
                text=segments_to_predict[idx * 2],
                start_idx=0,
                end_idx=split_idx,
                register_probs=all_probs[idx * 2].tolist(),
                embedding=all_embeddings[idx * 2].tolist(),
            )
            seg2 = Segment(
                text=segments_to_predict[idx * 2 + 1],
                start_idx=split_idx,
                end_idx=total_sentences,
                register_probs=all_probs[idx * 2 + 1].tolist(),
                embedding=all_embeddings[idx * 2 + 1].tolist(),
            )
            valid_segmentations.append((seg1, seg2))

        return valid_segmentations

    def evaluate_split(
        self,
        original_registers: Set[str],
        original_probs: np.ndarray,
        segment1: Segment,
        segment2: Segment,
    ) -> float:
        """Evaluate split and return score (entropy reduction)"""
        registers1 = self.get_dominant_registers(np.array(segment1.register_probs))
        registers2 = self.get_dominant_registers(np.array(segment2.register_probs))

        if not (registers1 or registers2):
            return float("-inf")

        if registers1 == registers2:
            return float("-inf")

        preserved_registers = registers1.union(registers2)
        if not original_registers.issubset(preserved_registers):
            return float("-inf")

        entropy1 = self.compute_entropy(np.array(segment1.register_probs))
        entropy2 = self.compute_entropy(np.array(segment2.register_probs))
        avg_entropy = (entropy1 + entropy2) / 2

        return self.compute_entropy(original_probs) - avg_entropy

    def combine_segments(self, segments: List[Segment]) -> List[Segment]:
        """Combine adjacent segments with the same registers with batched predictions"""
        if not segments:
            return segments

        # First pass: identify segments to combine
        combined = []
        current_segment = segments[0]
        current_registers = self.get_dominant_registers(
            np.array(current_segment.register_probs)
        )
        segments_to_predict = []

        for next_segment in segments[1:]:
            next_registers = self.get_dominant_registers(
                np.array(next_segment.register_probs)
            )

            if current_registers == next_registers:
                current_segment = Segment(
                    text=current_segment.text + " " + next_segment.text,
                    start_idx=current_segment.start_idx,
                    end_idx=next_segment.end_idx,
                    register_probs=[],
                    embedding=[],
                )
            else:
                segments_to_predict.append(current_segment)
                current_segment = next_segment
                current_registers = next_registers

        segments_to_predict.append(current_segment)

        # Predict all segments in batches
        texts_to_predict = [seg.text for seg in segments_to_predict]
        all_probs = []
        all_embeddings = []

        for i in range(0, len(texts_to_predict), self.batch_size):
            batch_texts = texts_to_predict[i : i + self.batch_size]
            probs, embeddings = self.predict_batch(batch_texts)
            all_probs.extend(probs)
            all_embeddings.extend(embeddings)

        # Update segments with predictions
        final_segments = []
        for idx, segment in enumerate(segments_to_predict):
            segment.register_probs = all_probs[idx].tolist()
            segment.embedding = all_embeddings[idx].tolist()
            final_segments.append(segment)

        return final_segments

    def segment_recursively(self, text: str) -> List[Segment]:
        """Recursively segment text based on register consistency"""
        sentences = self.split_to_sentences(text)
        if len(sentences) < 2 or len(text) < self.min_segment_length * 2:
            probs, emb = self.predict_registers_and_embeddings(text)
            return [
                Segment(
                    text=text,
                    start_idx=0,
                    end_idx=len(sentences),
                    register_probs=probs.tolist(),
                    embedding=emb.tolist(),
                )
            ]

        original_probs, _ = self.predict_registers_and_embeddings(text)
        original_registers = self.get_dominant_registers(original_probs)

        valid_segmentations = self.get_valid_segmentations(sentences)
        if not valid_segmentations:
            probs, emb = self.predict_registers_and_embeddings(text)
            return [
                Segment(
                    text=text,
                    start_idx=0,
                    end_idx=len(sentences),
                    register_probs=probs.tolist(),
                    embedding=emb.tolist(),
                )
            ]

        best_score = float("-inf")
        best_segmentation = None

        for seg1, seg2 in valid_segmentations:
            score = self.evaluate_split(original_registers, original_probs, seg1, seg2)
            if score > best_score:
                best_score = score
                best_segmentation = (seg1, seg2)

        if best_score <= 0 or best_segmentation is None:
            probs, emb = self.predict_registers_and_embeddings(text)
            return [
                Segment(
                    text=text,
                    start_idx=0,
                    end_idx=len(sentences),
                    register_probs=probs.tolist(),
                    embedding=emb.tolist(),
                )
            ]

        final_segments = []
        seg1, seg2 = best_segmentation
        final_segments.extend(self.segment_recursively(seg1.text))
        final_segments.extend(self.segment_recursively(seg2.text))
        return final_segments


def format_segments(
    segments: List[Segment],
    doc_id: str,
    id2label: Dict[int, str],
    threshold: float = 0.4,
) -> str:
    """Format segments for pretty printing."""
    output = [f"Text [{doc_id}]"]

    for i, segment in enumerate(segments, 1):
        output.append("---")
        # Get register labels above threshold
        labels = [
            id2label[i]
            for i, prob in enumerate(segment.register_probs)
            if prob > threshold
        ]
        output.append(f"Segment {i}: [{', '.join(labels)}]")
        display_text = (
            segment.text[:10000] + "..." if len(segment.text) > 10000 else segment.text
        )
        output.append(f"Text: {display_text}")

    output.append("---------------------")
    return "\n".join(output)


def process_file(
    input_path: str,
    output_path: str = None,
    min_segment_length: int = 300,
    register_threshold: float = 0.4,
    batch_size: int = 32,
    print_only: bool = False,
):
    """Process input JSONL file and write segmented output"""
    segmenter = RegisterSegmenter(
        min_segment_length=min_segment_length,
        register_threshold=register_threshold,
        batch_size=batch_size,
    )

    with open(input_path, "r", encoding="utf-8") as fin:
        fout = open(output_path, "w", encoding="utf-8") if output_path else None

        try:
            for line_num, line in enumerate(fin, 1):
                data = json.loads(line.strip())
                text = data["text"]

                # Get initial segmentation
                segments = segmenter.segment_recursively(text)

                # Combine segments with same registers and re-predict
                segments = segmenter.combine_segments(segments)

                if print_only:
                    print(
                        format_segments(
                            segments,
                            f"DOC_{line_num}",
                            segmenter.id2label,
                            segmenter.register_threshold,
                        )
                    )
                else:
                    data["segmentation"] = {
                        "texts": [seg.text for seg in segments],
                        "register_probabilities": [
                            seg.register_probs for seg in segments
                        ],
                        "embeddings": [seg.embedding for seg in segments],
                    }
                    fout.write(json.dumps(data, ensure_ascii=False) + "\n")
                    fout.flush()

        finally:
            if fout:
                fout.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="Input JSONL file")
    parser.add_argument(
        "output_file",
        nargs="?",
        help="Output JSONL file (optional if --print-only is used)",
    )
    parser.add_argument(
        "--min-length", type=int, default=300, help="Minimum segment length"
    )
    parser.add_argument(
        "--register-threshold",
        type=float,
        default=0.4,
        help="Threshold for considering a register as dominant",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for predictions"
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Print segments instead of saving to file",
    )
    args = parser.parse_args()

    if args.print_only and args.output_file:
        parser.error("Cannot specify output file when using --print-only")

    process_file(
        args.input_file,
        args.output_file,  # Fixed from output_path to output_file
        args.min_length,
        args.register_threshold,
        args.batch_size,
        args.print_only,
    )
