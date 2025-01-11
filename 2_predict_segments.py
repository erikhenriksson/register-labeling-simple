import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import spacy
import torch
import torch.nn.functional as F
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
    embedding: List[float]
    register_probs: List[float]


class TextSegmenter:
    def __init__(self, min_segment_length: int = 300):
        # Load spaCy for sentence splitting
        self.nlp = spacy.load("xx_ent_wiki_sm")
        if "sentencizer" not in self.nlp.pipe_names:
            self.nlp.add_pipe("sentencizer")

        self.min_segment_length = min_segment_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")
        self.model = self.model.to(self.device)
        self.model.eval()
        self.model.config.output_hidden_states = True
        self.id2label = self.model.config.id2label

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

    def compute_cohesion(self, embeddings: np.ndarray) -> float:
        """Compute average cosine similarity within a segment"""
        if len(embeddings) < 2:
            return 1.0

        # Normalize embeddings for faster cosine computation
        normalized = embeddings / np.linalg.norm(embeddings, axis=1)[:, np.newaxis]
        # Compute all pairwise similarities at once
        similarities = np.dot(normalized, normalized.T)
        # Get upper triangle excluding diagonal
        mask = np.triu(np.ones_like(similarities), k=1).astype(bool)
        return float(np.mean(similarities[mask]))

    def compute_dissimilarity(
        self, embeddings1: np.ndarray, embeddings2: np.ndarray
    ) -> float:
        """Compute average cosine dissimilarity between two segments"""
        # Normalize embeddings
        norm1 = embeddings1 / np.linalg.norm(embeddings1, axis=1)[:, np.newaxis]
        norm2 = embeddings2 / np.linalg.norm(embeddings2, axis=1)[:, np.newaxis]
        # Compute all similarities at once
        similarities = np.dot(norm1, norm2.T)
        return float(1 - np.mean(similarities))

    def evaluate_segmentation(
        self, sentence_embeddings: List[np.ndarray], segments: List[Segment]
    ) -> float:
        """
        Evaluate segmentation quality comparing ratio of dissimilarity to cohesion.
        Returns positive score only when segments are more different from each other
        than they are internally cohesive.
        """
        embeddings_list = [
            np.array(
                [sentence_embeddings[i] for i in range(seg.start_idx, seg.end_idx)]
            )
            for seg in segments
        ]

        # Compute cohesion for each segment
        cohesions = [self.compute_cohesion(emb) for emb in embeddings_list]
        avg_cohesion = np.mean(cohesions)
        print(f"Cohesion per segment: {cohesions}")
        print(f"Average cohesion: {avg_cohesion}")

        # Compute dissimilarity between all segment pairs
        dissimilarities = []
        for i in range(len(embeddings_list)):
            for j in range(i + 1, len(embeddings_list)):
                diss = self.compute_dissimilarity(
                    embeddings_list[i], embeddings_list[j]
                )
                dissimilarities.append(diss)
                print(f"Dissimilarity between segments {i} and {j}: {diss}")

        avg_dissimilarity = np.mean(dissimilarities)
        print(f"Average dissimilarity: {avg_dissimilarity}")

        # Score using ratio of dissimilarity to cohesion
        # Subtract 1 so positive score means dissimilarity > cohesion
        score = avg_dissimilarity / avg_cohesion - 1.0

        print(f"Final score (dissimilarity/cohesion - 1): {score}")
        print("----------------------------------------")

        return score

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
        """Recursively segment text until no valid segmentations remain"""
        # Split into sentences and precompute embeddings once
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

        # Precompute embeddings for all sentences
        sentence_embeddings, sentence_probs = self.precompute_embeddings(sentences)

        # Get all valid segmentations
        valid_segmentations = self.get_valid_segmentations(
            sentences, sentence_embeddings, sentence_probs
        )
        if not valid_segmentations:
            return [
                Segment(
                    text=text,
                    start_idx=0,
                    end_idx=len(sentences),
                    embedding=np.mean(sentence_embeddings, axis=0).tolist(),
                    register_probs=np.mean(sentence_probs, axis=0).tolist(),
                )
            ]

        # Find best segmentation, but only if it improves over no split
        best_score = 0  # Changed from -float("inf") since we only want positive scores
        best_segmentation = None
        for segmentation in valid_segmentations:
            score = self.evaluate_segmentation(sentence_embeddings, segmentation)
            if score > best_score:  # Will only update if score is positive
                best_score = score
                best_segmentation = segmentation

        # If no positive scoring segmentation found, return single segment
        if best_segmentation is None:
            return [
                Segment(
                    text=text,
                    start_idx=0,
                    end_idx=len(sentences),
                    embedding=np.mean(sentence_embeddings, axis=0).tolist(),
                    register_probs=np.mean(sentence_probs, axis=0).tolist(),
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
            segment.text[:10000] + "..." if len(segment.text) > 100000 else segment.text
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
    segmenter = TextSegmenter(min_segment_length=min_segment_length)

    with open(input_path, "r", encoding="utf-8") as fin:
        # Open output file only if not in print_only mode
        fout = open(output_path, "w", encoding="utf-8") if output_path else None

        try:
            for line_num, line in enumerate(fin, 1):
                # Parse input JSON
                data = json.loads(line.strip())
                text = data["text"]

                # Perform hierarchical segmentation
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
                        "embeddings": [seg.embedding for seg in segments],
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
