import json
import os
from itertools import combinations
from typing import List, Dict, Tuple
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import spacy
from tqdm import tqdm
import numpy as np

# Set environment variables
os.environ["HF_HOME"] = ".hf/hf_home"
os.environ["XDG_CACHE_HOME"] = ".hf/xdg_cache_home"


class TextSegmenter:
    def __init__(
        self, model_path: str, min_segment_chars: int = 300, prob_threshold: float = 0.5
    ):
        self.min_segment_chars = min_segment_chars
        self.prob_threshold = prob_threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load model and tokenizer
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")
        self.model = self.model.to(self.device)
        self.model.eval()
        self.model.config.output_hidden_states = True

        # Load spaCy for sentence splitting
        self.nlp = spacy.load("xx_ent_wiki_sm")
        if "sentencizer" not in self.nlp.pipe_names:
            self.nlp.add_pipe("sentencizer")

    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using spaCy."""
        doc = self.nlp(text)
        return [str(sent).strip() for sent in doc.sents if str(sent).strip()]

    def get_prediction(self, text: str) -> Tuple[List[float], List[float]]:
        """Get register probabilities and embedding for a text segment."""
        inputs = self.tokenizer(
            text, padding=True, truncation=True, max_length=2048, return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = F.sigmoid(outputs.logits)[0]
            embedding = outputs.hidden_states[-1][0, 0, :]

        return ([round(float(p), 3) for p in probs], [float(x) for x in embedding])

    def get_valid_breakpoints(self, sentences: List[str]) -> List[List[int]]:
        """Generate valid segmentations respecting minimum segment length."""
        n = len(sentences)
        valid_breakpoints = []

        # Helper function to check segment lengths
        def is_valid_segmentation(breaks: List[int]) -> bool:
            start = 0
            breaks = breaks + [n]

            for end in breaks:
                segment = " ".join(sentences[start:end])
                if len(segment) < self.min_segment_chars:
                    return False
                start = end
            return True

        # Generate and filter valid breakpoint combinations
        for k in range(n):
            for breakpoints in combinations(range(1, n), k):
                if is_valid_segmentation(list(breakpoints)):
                    valid_breakpoints.append(list(breakpoints))

        return valid_breakpoints

    def should_merge_segments(self, probs1: List[float], probs2: List[float]) -> bool:
        """Determine if segments should be merged based on multi-label predictions."""
        # Convert probabilities to binary labels using threshold
        labels1 = [1 if p >= self.prob_threshold else 0 for p in probs1]
        labels2 = [1 if p >= self.prob_threshold else 0 for p in probs2]

        # Compare binary label vectors
        return labels1 == labels2

    def evaluate_segmentation(
        self, sentences: List[str], breakpoints: List[int], cache: Dict[str, tuple]
    ) -> Tuple[float, List[str], List[List[float]], List[List[float]]]:
        """Evaluate a segmentation and return score, segments, probabilities, and embeddings."""
        segments = []
        start = 0
        probs_list = []
        emb_list = []
        total_score = 0

        breakpoints = breakpoints + [len(sentences)]

        for end in breakpoints:
            segment = " ".join(sentences[start:end])

            if segment in cache:
                probs, emb = cache[segment]
            else:
                probs, emb = self.get_prediction(segment)
                cache[segment] = (probs, emb)

            segments.append(segment)
            probs_list.append(probs)
            emb_list.append(emb)
            total_score += sum(p for p in probs if p >= self.prob_threshold)

            start = end

        return total_score, segments, probs_list, emb_list

    def get_active_registers(self, probs: List[float]) -> List[str]:
        """Get list of active registers based on probability threshold."""
        active = []
        for idx, prob in enumerate(probs):
            if prob >= self.prob_threshold:
                active.append(self.model.config.id2label[idx])
        return active

    def print_segments(self, segments: List[str], probs_list: List[List[float]]):
        """Print segments with their register labels."""
        print("\n=== Document Segmentation ===")
        for i, (segment, probs) in enumerate(zip(segments, probs_list), 1):
            registers = self.get_active_registers(probs)
            print(f"\nSegment {i}: {registers}")
            print(f"Text: {segment[:100]}...")  # Print first 100 chars
            print("---")

    def process_document(self, text: str) -> Dict:
        """Process a single document and find optimal segmentation."""
        sentences = self.split_into_sentences(text)
        prediction_cache = {}

        valid_segmentations = self.get_valid_breakpoints(sentences)

        best_score = -1
        best_segments = []
        best_probs = []
        best_embeddings = []

        for breakpoints in valid_segmentations:
            score, segments, probs, embs = self.evaluate_segmentation(
                sentences, breakpoints, prediction_cache
            )

            if score > best_score:
                best_score = score
                best_segments = segments
                best_probs = probs
                best_embeddings = embs

        # Merge consecutive segments with same register patterns
        final_segments = []
        final_probs = []
        final_embeddings = []

        if best_segments:
            current_segment = best_segments[0]
            current_probs = best_probs[0]
            current_emb = best_embeddings[0]

            for i in range(1, len(best_segments)):
                if self.should_merge_segments(current_probs, best_probs[i]):
                    # Same register pattern, merge segments
                    current_segment += " " + best_segments[i]
                    # Update probabilities and embeddings (take max of probabilities)
                    current_probs = [
                        max(a, b) for a, b in zip(current_probs, best_probs[i])
                    ]
                    # For embeddings, we could take mean or keep the first one
                    current_emb = [
                        0.5 * (a + b) for a, b in zip(current_emb, best_embeddings[i])
                    ]
                else:
                    # Different register pattern, add current and start new
                    final_segments.append(current_segment)
                    final_probs.append(current_probs)
                    final_embeddings.append(current_emb)
                    current_segment = best_segments[i]
                    current_probs = best_probs[i]
                    current_emb = best_embeddings[i]

            # Add last segment
            final_segments.append(current_segment)
            final_probs.append(current_probs)
            final_embeddings.append(current_emb)

        # Print segment information
        self.print_segments(final_segments, final_probs)

        return {
            "texts": final_segments,
            "register_probabilities": final_probs,
            "embeddings": final_embeddings,
            "register_labels": [
                self.get_active_registers(probs) for probs in final_probs
            ],
        }


def main():
    import sys

    if len(sys.argv) != 3:
        print("Usage: python script.py input.jsonl output.jsonl")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    # Initialize segmenter
    segmenter = TextSegmenter(
        model_path="/scratch/project_2011770/bge-2048",
        min_segment_chars=300,
        prob_threshold=0.5,
    )

    # Process file
    with open(input_file, "r") as fin, open(output_file, "w") as fout:
        for line in tqdm(fin):
            data = json.loads(line.strip())
            # Process the document
            segmentation = segmenter.process_document(data["text"])
            # Add segmentation to data
            data["segmentation"] = segmentation
            # Write to output
            fout.write(json.dumps(data, ensure_ascii=False) + "\n")
            fout.flush()


if __name__ == "__main__":
    main()
