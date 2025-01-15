import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import spacy
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from scipy.stats import entropy

# Set environment variables for HuggingFace
os.environ["HF_HOME"] = ".hf/hf_home"
os.environ["XDG_CACHE_HOME"] = ".hf/xdg_cache_home"

model_path = "/scratch/project_2011770/bge-2048"


@dataclass
class Segment:
    text: str
    start_idx: int
    end_idx: int
    embedding: List[float]
    register_probs: List[float]


class TextSegmenter:
    def __init__(self, min_segment_length: int = 300, min_gain_threshold: float = 0.05):
        # Load spaCy for sentence splitting
        self.nlp = spacy.load("xx_ent_wiki_sm")
        if "sentencizer" not in self.nlp.pipe_names:
            self.nlp.add_pipe("sentencizer")

        self.min_segment_length = min_segment_length
        self.min_gain_threshold = min_gain_threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")
        self.model = self.model.to(self.device)
        self.model.eval()
        self.model.config.output_hidden_states = True
        self.id2label = self.model.config.id2label

    def compute_l2_similarity(
        self, embeddings1: np.ndarray, embeddings2: np.ndarray
    ) -> float:
        """
        Compute similarity between two sets of embeddings using L2 distance.
        Returns a similarity score between 0 and 1, where 1 means identical.
        """
        distances = np.zeros((len(embeddings1), len(embeddings2)))
        for i in range(len(embeddings1)):
            for j in range(len(embeddings2)):
                l2_dist = np.linalg.norm(embeddings1[i] - embeddings2[j])
                distances[i, j] = np.exp(-l2_dist)

        return float(np.mean(distances))

    def compute_segment_cohesion(self, embeddings: np.ndarray) -> float:
        """Compute average L2-based similarity within a segment"""
        if len(embeddings) < 2:
            return 1.0

        similarities = np.zeros((len(embeddings), len(embeddings)))
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                l2_dist = np.linalg.norm(embeddings[i] - embeddings[j])
                similarity = np.exp(-l2_dist)
                similarities[i, j] = similarity
                similarities[j, i] = similarity

        mask = np.triu(np.ones_like(similarities), k=1).astype(bool)
        return float(np.mean(similarities[mask]))

    def compute_segment_dissimilarity(
        self, embeddings1: np.ndarray, embeddings2: np.ndarray
    ) -> float:
        """Compute average L2-based dissimilarity between two segments"""
        similarity = self.compute_l2_similarity(embeddings1, embeddings2)
        return 1.0 - similarity

    def precompute_embeddings(self, sentences: List[str]):
        """Precompute embeddings for all sentences in batches"""
        embeddings = []
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
                batch_embeddings = outputs.hidden_states[-1][:, 0, :].cpu().numpy()
                batch_probs = torch.sigmoid(outputs.logits).cpu().numpy()

                embeddings.extend(batch_embeddings)
                probs.extend(batch_probs)

        return embeddings, probs

    def split_to_sentences(self, text: str) -> List[str]:
        """Split text into sentences using spaCy"""
        doc = self.nlp(text)
        return [sent.text.strip() for sent in doc.sents]

    def compute_register_gain(
        self, parent_probs: np.ndarray, child_segments: List[Segment]
    ) -> float:
        """
        Compute register gain based on how focused each child segment becomes.
        A segment is better if it has clearer dominant registers than its parent.
        """
        # Normalize parent probabilities
        parent_probs = np.array(parent_probs) / np.sum(parent_probs)

        # Get the highest register probability in parent
        parent_focus = np.max(parent_probs)

        # For each child, compute how much more focused its registers are
        child_gains = []
        for seg in child_segments:
            child_probs = np.array(seg.register_probs)
            child_probs = child_probs / np.sum(child_probs)
            child_focus = np.max(child_probs)
            # Gain is positive if child is more focused than parent
            child_gains.append(child_focus - parent_focus)

        # Average the gains across children
        return np.mean(child_gains)

    def compute_information_gain(
        self, parent_segment: Segment, child_segments: List[Segment]
    ) -> float:
        """
        Compute information gain from splitting a segment into subsegments.
        Uses both register focus and semantic cohesion.
        """
        # Compute how much more focused the registers become
        register_gain = self.compute_register_gain(
            parent_segment.register_probs, child_segments
        )
        print(f"Register gain: {register_gain}")

        # Compute semantic coherence improvement
        parent_cohesion = self.compute_segment_cohesion(
            np.array(parent_segment.embedding).reshape(1, -1)
        )
        child_cohesions = [
            self.compute_segment_cohesion(np.array(seg.embedding).reshape(1, -1))
            for seg in child_segments
        ]
        cohesion_gain = np.mean(child_cohesions) - parent_cohesion
        print(f"Cohesion gain: {cohesion_gain}")

        # Combine both metrics (70% register, 30% cohesion)
        total_gain = 0.7 * register_gain + 0.3 * cohesion_gain
        print(f"Total gain: {total_gain}")
        print("-" * 50)

        return total_gain

        # Scale the gains to be positive when desirable
        register_gain = -register_gain  # Now positive when children have lower entropy
        cohesion_gain = cohesion_gain  # Already positive when children more coherent
        print(f"Cohesion gain: {cohesion_gain}")

        # Combine both metrics (70% register, 30% cohesion)
        total_gain = 0.7 * register_gain + 0.3 * cohesion_gain
        print(f"Total gain: {total_gain}")
        print("-" * 50)

        return total_gain

    def should_split(
        self, parent_segment: Segment, potential_splits: List[List[Segment]]
    ) -> Tuple[bool, List[Segment]]:
        """
        Determine if splitting is beneficial based on information gain.
        Returns (should_split, best_split)
        """
        best_gain = -float("inf")
        best_split = None

        for split in potential_splits:
            gain = self.compute_information_gain(parent_segment, split)
            if gain > best_gain:
                best_gain = gain
                best_split = split

        return best_gain > self.min_gain_threshold, best_split

    def get_valid_segmentations(
        self,
        sentences: List[str],
        sentence_embeddings: List[np.ndarray],
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
                    embedding=np.mean(sentence_embeddings[:split_idx], axis=0).tolist(),
                    register_probs=np.mean(sentence_probs[:split_idx], axis=0).tolist(),
                )
                seg2 = Segment(
                    text=segment2,
                    start_idx=split_idx,
                    end_idx=total_sentences,
                    embedding=np.mean(sentence_embeddings[split_idx:], axis=0).tolist(),
                    register_probs=np.mean(sentence_probs[split_idx:], axis=0).tolist(),
                )
                valid_segmentations.append([seg1, seg2])

        return valid_segmentations

    def segment_recursively(self, text: str) -> List[Segment]:
        """Modified recursive segmentation with information gain criterion"""
        sentences = self.split_to_sentences(text)
        if len(sentences) < 2 or len(text) < self.min_segment_length * 2:
            embedding, probs = self.precompute_embeddings([text])
            return [
                Segment(
                    text=text,
                    start_idx=0,
                    end_idx=len(sentences),
                    embedding=embedding[0].tolist(),
                    register_probs=probs[0].tolist(),
                )
            ]

        # Create parent segment
        parent_embedding, parent_probs = self.precompute_embeddings([text])
        parent_segment = Segment(
            text=text,
            start_idx=0,
            end_idx=len(sentences),
            embedding=parent_embedding[0].tolist(),
            register_probs=parent_probs[0].tolist(),
        )

        # Get potential splits
        sentence_embeddings, sentence_probs = self.precompute_embeddings(sentences)
        valid_segmentations = self.get_valid_segmentations(
            sentences, sentence_embeddings, sentence_probs
        )

        if not valid_segmentations:
            return [parent_segment]

        # Check if we should split based on information gain
        should_split, best_split = self.should_split(
            parent_segment, valid_segmentations
        )

        if not should_split:
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
    min_gain_threshold: float = 0.15,
    print_only: bool = False,
):
    """Process input JSONL file and write segmented output."""
    segmenter = TextSegmenter(
        min_segment_length=min_segment_length, min_gain_threshold=min_gain_threshold
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
        "--min-gain",
        type=float,
        default=0.15,
        help="Minimum information gain threshold for splitting",
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
