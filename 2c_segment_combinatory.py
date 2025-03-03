import json
import os
from itertools import combinations
from typing import List, Dict, Tuple
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import spacy
from tqdm import tqdm

# Set environment variables
os.environ["HF_HOME"] = ".hf/hf_home"
os.environ["XDG_CACHE_HOME"] = ".hf/xdg_cache_home"


class TextSegmenter:
    def __init__(
        self,
        model_path: str,
        prob_threshold: float = 0.5,
        initial_min_chars: int = 300,
        max_groups: int = 20,
    ):
        self.prob_threshold = prob_threshold
        self.initial_min_chars = initial_min_chars
        self.max_groups = max_groups

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

    def combine_short_sentences(self, sentences: List[str]) -> List[str]:
        """Combine sentences into larger blocks until we have at most max_groups blocks."""
        min_chars = self.initial_min_chars

        while True:
            result = []
            buffer = ""

            for i, sentence in enumerate(sentences):
                if len(sentence) >= min_chars:
                    if buffer:
                        result.append(buffer.strip())
                        buffer = ""
                    result.append(sentence)
                else:
                    buffer += (buffer and " ") + sentence

                    if len(buffer) >= min_chars:
                        result.append(buffer.strip())
                        buffer = ""

            if buffer:
                result.append(buffer.strip())

            i = 0
            while i < len(result):
                if len(result[i]) < min_chars:
                    if i < len(result) - 1:
                        result[i + 1] = result[i] + " " + result[i + 1]
                        result.pop(i)
                    elif i > 0:
                        result[i - 1] += " " + result[i]
                        result.pop(i)
                    else:
                        break
                else:
                    i += 1

            if len(result) <= self.max_groups:
                return result
            min_chars += 1

    def batch_prediction(
        self, texts: List[str]
    ) -> Tuple[List[List[float]], List[List[float]]]:
        """Process multiple texts in batches."""
        all_probs = []
        all_embs = []

        for i in range(0, len(texts), 16):  # Batch size of 16
            batch = texts[i : i + 16]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=2048,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = F.sigmoid(outputs.logits)
                embeddings = outputs.hidden_states[-1][:, 0, :]

            all_probs.extend([[round(float(p), 3) for p in row] for row in probs])
            all_embs.extend([[float(x) for x in row] for row in embeddings])

        return all_probs, all_embs

    def evaluate_segmentation(
        self, blocks: List[str], breakpoints: List[int], cache: Dict[str, tuple]
    ) -> Tuple[float, List[str], List[List[float]], List[List[float]]]:
        """Evaluate a segmentation and return score, segments, probabilities, and embeddings."""
        segments = []
        uncached_segments = []
        uncached_positions = []

        # Generate all segments and identify which need prediction
        start = 0
        breakpoints = breakpoints + [len(blocks)]

        for end in breakpoints:
            segment = " ".join(blocks[start:end])
            segments.append(segment)

            if segment not in cache:
                uncached_segments.append(segment)
                uncached_positions.append(len(segments) - 1)

            start = end

        # Batch predict uncached segments
        if uncached_segments:
            batch_probs, batch_embs = self.batch_prediction(uncached_segments)
            # Update cache with new predictions
            for i, segment in enumerate(uncached_segments):
                cache[segment] = (batch_probs[i], batch_embs[i])

        # Collect all predictions (from cache)
        probs_list = []
        emb_list = []
        total_score = 0

        for segment in segments:
            probs, emb = cache[segment]
            probs_list.append(probs)
            emb_list.append(emb)
            total_score += max(probs)

        # Calculate average score
        total_score /= len(segments)

        return total_score, segments, probs_list, emb_list

    def should_merge_segments(self, probs1: List[float], probs2: List[float]) -> bool:
        """Determine if segments should be merged based on multi-label predictions."""
        labels1 = [1 if p >= self.prob_threshold else 0 for p in probs1]
        labels2 = [1 if p >= self.prob_threshold else 0 for p in probs2]
        return labels1 == labels2

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
            print(f"Text: {segment[:100]}...")
            print("---")

    def process_document(self, text: str) -> Dict:
        """Process a single document and find optimal segmentation."""
        # First split into sentences and combine into blocks
        sentences = self.split_into_sentences(text)
        blocks = self.combine_short_sentences(sentences)
        print(f"\nNumber of blocks after combining: {len(blocks)}")

        # Generate all possible breakpoint combinations
        breakpoint_combinations = []
        for k in range(len(blocks)):
            for breakpoints in combinations(range(1, len(blocks)), k):
                breakpoint_combinations.append(list(breakpoints))
        print(f"Number of possible segmentations: {len(breakpoint_combinations)}")

        # Find best segmentation
        prediction_cache = {}
        best_score = -1
        best_segments = []
        best_probs = []
        best_embeddings = []

        for breakpoints in breakpoint_combinations:
            score, segments, probs, embs = self.evaluate_segmentation(
                blocks, breakpoints, prediction_cache
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
                    # Same register pattern, merge segments and get new predictions
                    current_segment += " " + best_segments[i]
                    batch_probs, batch_embs = self.batch_prediction([current_segment])
                    current_probs, current_emb = batch_probs[0], batch_embs[0]
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
        prob_threshold=0.5,
        initial_min_chars=300,
        max_groups=20,
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
