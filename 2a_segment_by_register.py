import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
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


class EntropySegmenter:
    def __init__(self, min_segment_length: int = 300):
        # Load spaCy for sentence splitting
        self.nlp = spacy.load("xx_ent_wiki_sm")
        if "sentencizer" not in self.nlp.pipe_names:
            self.nlp.add_pipe("sentencizer")

        self.min_segment_length = min_segment_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model and tokenizer
        model_path = "/scratch/project_2011770/bge-2048"
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")
        self.model = self.model.to(self.device)
        self.model.eval()
        self.id2label = self.model.config.id2label

    def compute_register_probs(self, texts: List[str]) -> List[np.ndarray]:
        """Compute register probabilities for a batch of texts"""
        probs = []
        batch_size = 32

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            with torch.no_grad():
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=2048,
                    return_tensors="pt",
                ).to(self.device)
                outputs = self.model(**inputs)
                batch_probs = torch.sigmoid(outputs.logits).cpu().numpy()
                probs.extend(batch_probs)

        return probs

    def compute_entropy(self, probs: np.ndarray) -> float:
        """Compute entropy of register probability distribution"""
        # Add small epsilon to avoid log(0)
        eps = 1e-10
        probs_safe = np.clip(probs, eps, 1.0)
        return float(-np.sum(probs_safe * np.log(probs_safe)))

    def split_to_sentences(self, text: str) -> List[str]:
        """Split text into sentences using spaCy"""
        doc = self.nlp(text)
        return [sent.text.strip() for sent in doc.sents]

    def get_valid_segmentations(
        self,
        sentences: List[str],
        sentence_probs: List[np.ndarray],
    ) -> List[Tuple[Segment, Segment]]:
        """Get all valid segmentations based on minimum length constraint"""
        valid_segmentations = []
        total_sentences = len(sentences)

        for split_idx in range(1, total_sentences):
            segment1 = " ".join(sentences[:split_idx])
            segment2 = " ".join(sentences[split_idx:])

            if (
                len(segment1) >= self.min_segment_length
                and len(segment2) >= self.min_segment_length
            ):
                # Average probabilities for each segment
                probs1 = np.mean(sentence_probs[:split_idx], axis=0)
                probs2 = np.mean(sentence_probs[split_idx:], axis=0)

                seg1 = Segment(
                    text=segment1,
                    start_idx=0,
                    end_idx=split_idx,
                    register_probs=probs1.tolist(),
                )
                seg2 = Segment(
                    text=segment2,
                    start_idx=split_idx,
                    end_idx=total_sentences,
                    register_probs=probs2.tolist(),
                )
                valid_segmentations.append((seg1, seg2))

        return valid_segmentations

    def evaluate_split(
        self, original_entropy: float, segment1: Segment, segment2: Segment
    ) -> float:
        """
        Evaluate a potential split based on entropy improvement.
        Returns the improvement in average entropy (negative means improvement).
        """
        entropy1 = self.compute_entropy(np.array(segment1.register_probs))
        entropy2 = self.compute_entropy(np.array(segment2.register_probs))
        avg_entropy = (entropy1 + entropy2) / 2
        return avg_entropy - original_entropy

    def segment_recursively(self, text: str) -> List[Segment]:
        """Recursively segment text based on entropy improvement"""
        # Split into sentences and compute register probabilities
        sentences = self.split_to_sentences(text)
        if len(sentences) < 2 or len(text) < self.min_segment_length * 2:
            probs = self.compute_register_probs([text])[0]
            return [
                Segment(
                    text=text,
                    start_idx=0,
                    end_idx=len(sentences),
                    register_probs=probs.tolist(),
                )
            ]

        # Compute probabilities for all sentences
        sentence_probs = self.compute_register_probs(sentences)

        # Compute original entropy
        original_probs = np.mean(sentence_probs, axis=0)
        original_entropy = self.compute_entropy(original_probs)

        # Get all valid segmentations
        valid_segmentations = self.get_valid_segmentations(sentences, sentence_probs)
        if not valid_segmentations:
            return [
                Segment(
                    text=text,
                    start_idx=0,
                    end_idx=len(sentences),
                    register_probs=original_probs.tolist(),
                )
            ]

        # Find best segmentation that improves entropy
        best_improvement = 0
        best_segmentation = None

        for seg1, seg2 in valid_segmentations:
            improvement = -self.evaluate_split(original_entropy, seg1, seg2)
            if improvement > best_improvement:
                best_improvement = improvement
                best_segmentation = (seg1, seg2)

        # If no improvement found, return original segment
        if best_segmentation is None:
            return [
                Segment(
                    text=text,
                    start_idx=0,
                    end_idx=len(sentences),
                    register_probs=original_probs.tolist(),
                )
            ]

        # Recursively segment each part
        final_segments = []
        for segment in best_segmentation:
            final_segments.extend(self.segment_recursively(segment.text))

        return final_segments


def get_register_labels(
    probs: List[float], id2label: Dict[int, str], threshold: float = 0.5
) -> List[str]:
    """Convert probabilities to register labels using the model's id2label mapping."""
    return [id2label[i] for i, prob in enumerate(probs) if prob > threshold]


def format_segments(
    segments: List[Segment], doc_id: str, id2label: Dict[int, str]
) -> str:
    """Format segments for pretty printing."""
    output = [f"Text [{doc_id}]"]

    for i, segment in enumerate(segments, 1):
        output.append("---")
        labels = get_register_labels(segment.register_probs, id2label)
        output.append(f"Segment {i}: [{', '.join(labels)}]")
        # Truncate text if too long for display
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
    print_only: bool = False,
):
    """Process input JSONL file and write segmented output."""
    segmenter = EntropySegmenter(min_segment_length=min_segment_length)

    with open(input_path, "r", encoding="utf-8") as fin:
        # Open output file only if not in print_only mode
        fout = open(output_path, "w", encoding="utf-8") if output_path else None

        try:
            for line_num, line in enumerate(fin, 1):
                # Parse input JSON
                data = json.loads(line.strip())
                text = data["text"]

                # Perform entropy-based segmentation
                segments = segmenter.segment_recursively(text)

                if print_only:
                    # Print formatted segments
                    print(
                        format_segments(segments, f"DOC_{line_num}", segmenter.id2label)
                    )
                else:
                    # Add segmentation information to output
                    data["segmentation"] = {
                        "texts": [seg.text for seg in segments],
                        "register_probabilities": [
                            seg.register_probs for seg in segments
                        ],
                    }

                    # Write updated JSON to output file
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
        "--print-only",
        action="store_true",
        help="Print segments instead of saving to file",
    )
    args = parser.parse_args()

    if args.print_only and args.output_file:
        parser.error("Cannot specify output file when using --print-only")

    process_file(args.input_file, args.output_file, args.min_length, args.print_only)
