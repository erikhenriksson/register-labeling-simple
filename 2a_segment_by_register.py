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


class RegisterSegmenter:
    def __init__(self, min_segment_length: int = 300, register_threshold: float = 0.4):
        # Load spaCy for sentence splitting
        self.nlp = spacy.load("xx_ent_wiki_sm")
        if "sentencizer" not in self.nlp.pipe_names:
            self.nlp.add_pipe("sentencizer")

        self.min_segment_length = min_segment_length
        self.register_threshold = register_threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model and tokenizer
        model_path = "/scratch/project_2011770/bge-2048"
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")
        self.model = self.model.to(self.device)
        self.model.eval()
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

    def adjust_probs_for_hierarchy(self, probs: np.ndarray) -> np.ndarray:
        """
        Adjust probabilities to enforce hierarchical consistency:
        - If any child label is above threshold, set its parent to 0
        """
        adjusted_probs = probs.copy()

        # For each parent-children relationship
        for parent, children in self.labels_structure.items():
            if not children:  # Skip if no children
                continue

            # Get indices for parent and children
            parent_idx = self.label_to_idx[parent]
            children_indices = [self.label_to_idx[child] for child in children]

            # Check if any child is above threshold
            children_probs = adjusted_probs[children_indices]
            if any(prob >= self.register_threshold for prob in children_probs):
                # If yes, set parent probability to 0
                adjusted_probs[parent_idx] = 0.0

        return adjusted_probs

    def predict_registers(self, text: str) -> np.ndarray:
        """Predict register probabilities for a complete text segment"""
        with torch.no_grad():
            inputs = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=2048,
                return_tensors="pt",
            ).to(self.device)
            outputs = self.model(**inputs)
            probs = torch.sigmoid(outputs.logits)[0].cpu().numpy()
            # Adjust for hierarchy
            return self.adjust_probs_for_hierarchy(probs)

    def get_dominant_registers(self, probs: np.ndarray) -> Set[str]:
        """
        Get set of dominant registers (those above threshold).
        Considers hierarchical relationship when counting registers.
        """
        dominant = set()
        for i, prob in enumerate(probs):
            if prob >= self.register_threshold:
                label = self.id2label[i]
                # If it's a child label, add it but don't add its parent
                if label in self.child_to_parent:
                    dominant.add(label)
                else:
                    # If it's a parent label, add it only if none of its children are present
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
        """Get all valid segmentations based on minimum length constraint"""
        valid_segmentations = []
        total_sentences = len(sentences)

        for split_idx in range(1, total_sentences):
            segment1_text = " ".join(sentences[:split_idx])
            segment2_text = " ".join(sentences[split_idx:])

            if (
                len(segment1_text) >= self.min_segment_length
                and len(segment2_text) >= self.min_segment_length
            ):
                # Predict registers for complete segments
                probs1 = self.predict_registers(segment1_text)
                probs2 = self.predict_registers(segment2_text)

                seg1 = Segment(
                    text=segment1_text,
                    start_idx=0,
                    end_idx=split_idx,
                    register_probs=probs1.tolist(),
                )
                seg2 = Segment(
                    text=segment2_text,
                    start_idx=split_idx,
                    end_idx=total_sentences,
                    register_probs=probs2.tolist(),
                )
                valid_segmentations.append((seg1, seg2))

        return valid_segmentations

    def evaluate_split(
        self, original_registers: Set[str], segment1: Segment, segment2: Segment
    ) -> bool:
        """
        Evaluate if a split is beneficial based on register consistency.
        Returns True if split is beneficial, False otherwise.
        """
        # Get dominant registers for each segment
        registers1 = self.get_dominant_registers(np.array(segment1.register_probs))
        registers2 = self.get_dominant_registers(np.array(segment2.register_probs))

        # Split is beneficial if:
        # 1. At least one segment has a dominant register
        # 2. The registers are different between segments
        # 3. All original registers are preserved in at least one segment
        if not (registers1 or registers2):  # At least one must have a register
            return False

        # Check if the registers are different
        if registers1 == registers2:
            return False

        # Check if all original registers are preserved
        preserved_registers = registers1.union(registers2)
        if not original_registers.issubset(preserved_registers):
            return False

        return True

    def segment_recursively(self, text: str) -> List[Segment]:
        """Recursively segment text based on register consistency"""
        # Split into sentences
        sentences = self.split_to_sentences(text)
        if len(sentences) < 2 or len(text) < self.min_segment_length * 2:
            probs = self.predict_registers(text)
            return [
                Segment(
                    text=text,
                    start_idx=0,
                    end_idx=len(sentences),
                    register_probs=probs.tolist(),
                )
            ]

        # Get original registers from complete text
        original_probs = self.predict_registers(text)
        original_registers = self.get_dominant_registers(original_probs)

        # Get all valid segmentations
        valid_segmentations = self.get_valid_segmentations(sentences)
        if not valid_segmentations:
            return [
                Segment(
                    text=text,
                    start_idx=0,
                    end_idx=len(sentences),
                    register_probs=original_probs.tolist(),
                )
            ]

        # Find best segmentation that improves register consistency
        for seg1, seg2 in valid_segmentations:
            if self.evaluate_split(original_registers, seg1, seg2):
                # Get registers for each segment before recursing
                seg1_registers = self.get_dominant_registers(
                    np.array(seg1.register_probs)
                )
                seg2_registers = self.get_dominant_registers(
                    np.array(seg2.register_probs)
                )

                # Recursively segment each part, passing down their respective registers
                final_segments = []
                if seg1_registers:  # Only recurse if there are registers to preserve
                    final_segments.extend(self.segment_recursively(seg1.text))
                else:
                    final_segments.append(seg1)

                if seg2_registers:  # Only recurse if there are registers to preserve
                    final_segments.extend(self.segment_recursively(seg2.text))
                else:
                    final_segments.append(seg2)

                return final_segments

        # If no beneficial split found, return original segment
        return [
            Segment(
                text=text,
                start_idx=0,
                end_idx=len(sentences),
                register_probs=original_probs.tolist(),
            )
        ]


def get_register_labels(
    probs: List[float], id2label: Dict[int, str], threshold: float = 0.4
) -> List[str]:
    """Convert probabilities to register labels using the model's id2label mapping."""
    return [id2label[i] for i, prob in enumerate(probs) if prob > threshold]


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
        labels = get_register_labels(segment.register_probs, id2label, threshold)
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
    print_only: bool = False,
):
    """Process input JSONL file and write segmented output."""
    segmenter = RegisterSegmenter(
        min_segment_length=min_segment_length, register_threshold=register_threshold
    )

    with open(input_path, "r", encoding="utf-8") as fin:
        fout = open(output_path, "w", encoding="utf-8") if output_path else None

        try:
            for line_num, line in enumerate(fin, 1):
                data = json.loads(line.strip())
                text = data["text"]

                segments = segmenter.segment_recursively(text)

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
        "--print-only",
        action="store_true",
        help="Print segments instead of saving to file",
    )
    args = parser.parse_args()

    if args.print_only and args.output_file:
        parser.error("Cannot specify output file when using --print-only")

    process_file(
        args.input_file,
        args.output_file,
        args.min_length,
        args.register_threshold,
        args.print_only,
    )
