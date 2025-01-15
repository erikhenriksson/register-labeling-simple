import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import spacy
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Set environment variables for HuggingFace
os.environ["HF_HOME"] = ".hf/hf_home"
os.environ["XDG_CACHE_HOME"] = ".hf/xdg_cache_home"

model_path = "/scratch/project_2011770/bge-2048"


@dataclass
class Segment:
    text: str
    start_idx: int
    end_idx: int
    register_probs: List[float]


class TextSegmenter:
    def __init__(self, min_segment_length: int = 300, min_prob_gain: float = 0.1):
        # Load spaCy for sentence splitting
        self.nlp = spacy.load("xx_ent_wiki_sm")
        if "sentencizer" not in self.nlp.pipe_names:
            self.nlp.add_pipe("sentencizer")

        self.min_segment_length = min_segment_length
        self.min_prob_gain = min_prob_gain
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")
        self.model = self.model.to(self.device)
        self.model.eval()
        self.id2label = self.model.config.id2label

    def compute_register_probabilities(self, sentences: List[str]):
        """Compute register probabilities for sentences in batches"""
        probs = []
        batch_size = 32

        for i in range(0, len(sentences), batch_size):
            batch = sentences[i : i + batch_size]
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

    def split_to_sentences(self, text: str) -> List[str]:
        """Split text into sentences using spaCy"""
        doc = self.nlp(text)
        return [sent.text.strip() for sent in doc.sents]

    def compute_register_gain(
        self, parent_probs: np.ndarray, child_segments: List[Segment]
    ) -> float:
        """
        Compute register gain based on how focused each child segment becomes.
        Returns the average improvement in maximum register probability.
        """
        # Normalize parent probabilities
        parent_probs = np.array(parent_probs) / np.sum(parent_probs)
        parent_focus = np.max(parent_probs)
        print(f"Parent max probability: {parent_focus}")

        # For each child, compute how much more focused its registers are
        child_gains = []
        for seg in child_segments:
            child_probs = np.array(seg.register_probs)
            child_probs = child_probs / np.sum(child_probs)
            child_focus = np.max(child_probs)
            print(f"Child max probability: {child_focus}")
            child_gains.append(child_focus - parent_focus)

        avg_gain = np.mean(child_gains)
        print(f"Average gain: {avg_gain}")
        print("-" * 50)
        return avg_gain

    def get_valid_segmentations(
        self,
        sentences: List[str],
        sentence_probs: List[np.ndarray],
    ) -> List[List[Segment]]:
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
                seg1 = Segment(
                    text=segment1,
                    start_idx=0,
                    end_idx=split_idx,
                    register_probs=np.mean(sentence_probs[:split_idx], axis=0).tolist(),
                )
                seg2 = Segment(
                    text=segment2,
                    start_idx=split_idx,
                    end_idx=total_sentences,
                    register_probs=np.mean(sentence_probs[split_idx:], axis=0).tolist(),
                )
                valid_segmentations.append([seg1, seg2])

        return valid_segmentations

    def segment_recursively(self, text: str) -> List[Segment]:
        """Recursively segment text based on register probability gains"""
        sentences = self.split_to_sentences(text)
        if len(sentences) < 2 or len(text) < self.min_segment_length * 2:
            probs = self.compute_register_probabilities([text])
            return [
                Segment(
                    text=text,
                    start_idx=0,
                    end_idx=len(sentences),
                    register_probs=probs[0].tolist(),
                )
            ]

        # Create parent segment
        parent_probs = self.compute_register_probabilities([text])[0]
        parent_segment = Segment(
            text=text,
            start_idx=0,
            end_idx=len(sentences),
            register_probs=parent_probs.tolist(),
        )

        # Get potential splits
        sentence_probs = self.compute_register_probabilities(sentences)
        valid_segmentations = self.get_valid_segmentations(sentences, sentence_probs)

        if not valid_segmentations:
            return [parent_segment]

        # Find best segmentation based on register probability gain
        best_gain = -float("inf")
        best_split = None
        for split in valid_segmentations:
            gain = self.compute_register_gain(parent_probs, split)
            if gain > best_gain:
                best_gain = gain
                best_split = split

        # Only split if the gain exceeds our threshold
        if best_gain < self.min_prob_gain:
            return [parent_segment]

        # Recursively split each segment in the best split
        final_segments = []
        for segment in best_split:
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
        display_text = (
            segment.text[:10000] + "..." if len(segment.text) > 100000 else segment.text
        )
        output.append(f"Text: {display_text}")

    output.append("---------------------")
    return "\n".join(output)


def process_file(
    input_path: str,
    output_path: str = None,
    min_segment_length: int = 300,
    min_prob_gain: float = 0.1,
    print_only: bool = False,
):
    """Process input JSONL file and write segmented output."""
    segmenter = TextSegmenter(
        min_segment_length=min_segment_length, min_prob_gain=min_prob_gain
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
                        format_segments(segments, f"DOC_{line_num}", segmenter.id2label)
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
        "--min-gain",
        type=float,
        default=0.0,
        help="Minimum register probability gain threshold for splitting",
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
        args.min_gain,
        args.print_only,
    )
