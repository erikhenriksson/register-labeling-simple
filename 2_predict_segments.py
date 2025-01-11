import json
import spacy
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


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

        # Initialize model and tokenizer for register classification
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "TurkuNLP/web-register-classification-multilingual"
        )
        self.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")
        self.model = self.model.to(self.device)
        self.model.eval()

        # Enable hidden states output for embeddings
        self.model.config.output_hidden_states = True

    def get_embedding_and_probs(self, text: str) -> Tuple[List[float], List[float]]:
        """Get both embedding and register probabilities for a text segment."""
        with torch.no_grad():
            inputs = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=2048,
                return_tensors="pt",
            ).to(self.device)
            outputs = self.model(**inputs)

            # Get embeddings from the last hidden state
            embedding = outputs.hidden_states[-1][:, 0, :].cpu().numpy()[0]

            # Get probabilities using sigmoid for multilabel classification
            probs = F.sigmoid(outputs.logits).cpu().numpy()[0]

            return ([float(x) for x in embedding], [round(float(p), 3) for p in probs])

    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a text segment."""
        embedding, _ = self.get_embedding_and_probs(text)
        return embedding

    def get_register_probs(self, text: str) -> List[float]:
        """Get register probabilities for a text segment."""
        _, probs = self.get_embedding_and_probs(text)
        return probs

    def split_to_sentences(self, text: str) -> List[str]:
        """Split text into sentences using spaCy."""
        doc = self.nlp(text)
        return [sent.text.strip() for sent in doc.sents]

    def compute_cohesion(self, embeddings: List[List[float]]) -> float:
        """Compute average cosine similarity within a segment."""
        if len(embeddings) < 2:
            return 1.0

        embeddings = np.array(embeddings)
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                )
                similarities.append(sim)
        return float(np.mean(similarities))

    def compute_dissimilarity(
        self, embeddings1: List[List[float]], embeddings2: List[List[float]]
    ) -> float:
        """Compute average cosine dissimilarity between two segments."""
        embeddings1 = np.array(embeddings1)
        embeddings2 = np.array(embeddings2)

        similarities = []
        for emb1 in embeddings1:
            for emb2 in embeddings2:
                sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                similarities.append(sim)
        return float(1 - np.mean(similarities))

    def evaluate_segmentation(self, segments: List[Segment]) -> float:
        """Evaluate a segmentation based on cohesion and dissimilarity."""
        if len(segments) < 2:
            return 0.0

        # Compute average internal cohesion
        cohesions = []
        for segment in segments:
            sent_embeddings = [
                self.get_embedding(sent)
                for sent in self.split_to_sentences(segment.text)
            ]
            cohesions.append(self.compute_cohesion(sent_embeddings))
        avg_cohesion = np.mean(cohesions)

        # Compute average inter-segment dissimilarity
        dissimilarities = []
        for i in range(len(segments)):
            for j in range(i + 1, len(segments)):
                sent_embeddings1 = [
                    self.get_embedding(sent)
                    for sent in self.split_to_sentences(segments[i].text)
                ]
                sent_embeddings2 = [
                    self.get_embedding(sent)
                    for sent in self.split_to_sentences(segments[j].text)
                ]
                dissimilarities.append(
                    self.compute_dissimilarity(sent_embeddings1, sent_embeddings2)
                )
        avg_dissimilarity = np.mean(dissimilarities)

        # Combine metrics (equal weight)
        return 0.5 * avg_cohesion + 0.5 * avg_dissimilarity

    def get_valid_segmentations(self, sentences: List[str]) -> List[List[Segment]]:
        """Get all valid segmentations based on minimum length constraint."""
        valid_segmentations = []
        total_sentences = len(sentences)

        # Try different split points
        for split_idx in range(1, total_sentences):
            # Check all possible segment combinations
            segment1 = " ".join(sentences[:split_idx])
            segment2 = " ".join(sentences[split_idx:])

            if (
                len(segment1) >= self.min_segment_length
                and len(segment2) >= self.min_segment_length
            ):
                # Create segments with embeddings and probabilities
                seg1 = Segment(
                    text=segment1,
                    start_idx=0,
                    end_idx=split_idx,
                    embedding=self.get_embedding(segment1),
                    register_probs=self.get_register_probs(segment1),
                )
                seg2 = Segment(
                    text=segment2,
                    start_idx=split_idx,
                    end_idx=total_sentences,
                    embedding=self.get_embedding(segment2),
                    register_probs=self.get_register_probs(segment2),
                )
                valid_segmentations.append([seg1, seg2])

        return valid_segmentations

    def segment_recursively(self, text: str) -> List[Segment]:
        """Recursively segment text until no valid segmentations remain."""
        sentences = self.split_to_sentences(text)
        if len(sentences) < 2 or len(text) < self.min_segment_length * 2:
            # Base case: can't segment further
            return [
                Segment(
                    text=text,
                    start_idx=0,
                    end_idx=len(sentences),
                    embedding=self.get_embedding(text),
                    register_probs=self.get_register_probs(text),
                )
            ]

        # Get all valid segmentations
        valid_segmentations = self.get_valid_segmentations(sentences)
        if not valid_segmentations:
            return [
                Segment(
                    text=text,
                    start_idx=0,
                    end_idx=len(sentences),
                    embedding=self.get_embedding(text),
                    register_probs=self.get_register_probs(text),
                )
            ]

        # Evaluate all segmentations and choose the best
        best_score = -float("inf")
        best_segmentation = None
        for segmentation in valid_segmentations:
            score = self.evaluate_segmentation(segmentation)
            if score > best_score:
                best_score = score
                best_segmentation = segmentation

        # Recursively segment each part of the best segmentation
        final_segments = []
        for segment in best_segmentation:
            final_segments.extend(self.segment_recursively(segment.text))

        return final_segments


def process_file(input_path: str, output_path: str, min_segment_length: int = 300):
    """Process input JSONL file and write segmented output."""
    segmenter = TextSegmenter(min_segment_length=min_segment_length)

    with open(input_path, "r", encoding="utf-8") as fin, open(
        output_path, "w", encoding="utf-8"
    ) as fout:

        for line in fin:
            # Parse input JSON
            data = json.loads(line.strip())
            text = data["text"]

            # Perform hierarchical segmentation
            segments = segmenter.segment_recursively(text)

            # Add segmentation information to output
            data["segmentation"] = {
                "texts": [seg.text for seg in segments],
                "register_probabilities": [seg.register_probs for seg in segments],
                "embeddings": [seg.embedding for seg in segments],
            }

            # Write updated JSON to output file
            fout.write(json.dumps(data, ensure_ascii=False) + "\n")
            fout.flush()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="Input JSONL file")
    parser.add_argument("output_file", help="Output JSONL file")
    parser.add_argument(
        "--min-length", type=int, default=300, help="Minimum segment length"
    )
    args = parser.parse_args()

    process_file(args.input_file, args.output_file, args.min_length)
