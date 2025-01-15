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
        initial_min_words: int = 15,
        max_groups: int = 20,
    ):
        print("Initializing TextSegmenter...")
        self.prob_threshold = prob_threshold
        self.initial_min_words = initial_min_words
        self.max_groups = max_groups

        print("Setting up device...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load model and tokenizer
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")

        print(f"Loading model from {model_path}...")
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        print("Moving model to device...")
        self.model = self.model.to(self.device)
        print("Setting model to eval mode...")
        self.model.eval()
        self.model.config.output_hidden_states = True

        # Load spaCy
        print("Loading spaCy model...")
        self.nlp = spacy.load("xx_ent_wiki_sm")
        if "sentencizer" not in self.nlp.pipe_names:
            print("Adding sentencizer to spaCy pipeline...")
            self.nlp.add_pipe("sentencizer")

        print("Initialization complete.")

    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using spaCy."""
        doc = self.nlp(text)
        return [str(sent).strip() for sent in doc.sents if str(sent).strip()]

    def combine_short_sentences(self, sentences: List[str]) -> List[str]:
        """Combine sentences into larger blocks until we have at most max_groups blocks."""
        min_words = self.initial_min_words

        def count_words(sentence):
            return len(sentence.split())

        while True:
            result = []
            buffer = ""

            for i, sentence in enumerate(sentences):
                if count_words(sentence) >= min_words:
                    if buffer:
                        result.append(buffer.strip())
                        buffer = ""
                    result.append(sentence)
                else:
                    buffer += (buffer and " ") + sentence

                    # If the buffer reaches min_words, finalize it
                    if count_words(buffer) >= min_words:
                        result.append(buffer.strip())
                        buffer = ""

            # Handle leftover buffer
            if buffer:
                result.append(buffer.strip())

            # Final pass: Ensure no sentences in the result are below min_words
            i = 0
            while i < len(result):
                if count_words(result[i]) < min_words:
                    if i < len(result) - 1:  # Merge with the next sentence
                        result[i + 1] = result[i] + " " + result[i + 1]
                        result.pop(i)
                    elif i > 0:  # Merge with the previous sentence if it's the last one
                        result[i - 1] += " " + result[i]
                        result.pop(i)
                    else:  # Single short sentence case
                        break
                else:
                    i += 1

            if len(result) <= self.max_groups:
                return result
            min_words += 1

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

    def get_all_breakpoint_combinations(self, n_blocks: int) -> List[List[int]]:
        """Generate all possible breakpoint combinations."""
        all_breakpoints = []
        for k in range(n_blocks):
            for breakpoints in combinations(range(1, n_blocks), k):
                all_breakpoints.append(list(breakpoints))
        return all_breakpoints

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
            print(f"Text: {segment[:100]}...")  # Print first 100 chars
            print("---")

    def evaluate_segmentation(
        self, blocks: List[str], breakpoints: List[int], cache: Dict[str, tuple]
    ) -> Tuple[float, List[str], List[List[float]], List[List[float]]]:
        """Evaluate a segmentation and return score, segments, probabilities, and embeddings."""
        segments = []
        start = 0
        probs_list = []
        emb_list = []
        total_score = 0

        breakpoints = breakpoints + [len(blocks)]

        for end in breakpoints:
            segment = " ".join(blocks[start:end])

            if segment in cache:
                probs, emb = cache[segment]
            else:
                probs, emb = self.get_prediction(segment)
                cache[segment] = (probs, emb)

            segments.append(segment)
            probs_list.append(probs)
            emb_list.append(emb)
            total_score = total_score / len(
                breakpoints + [len(blocks)]
            )  # Use maximum probability as segment score

            start = end

        return total_score, segments, probs_list, emb_list

    def process_document(self, text: str) -> Dict:
        """Process a single document and find optimal segmentation."""
        # First split into sentences
        sentences = self.split_into_sentences(text)

        # Combine into larger blocks
        blocks = self.combine_short_sentences(sentences)
        print(f"\nNumber of blocks after combining: {len(blocks)}")

        # Generate all possible breakpoint combinations
        breakpoint_combinations = self.get_all_breakpoint_combinations(len(blocks))
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
                    # Same register pattern, merge segments
                    current_segment += " " + best_segments[i]
                    # Get new predictions for combined segment
                    current_probs, current_emb = self.get_prediction(current_segment)
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

    try:
        print("Starting segmenter initialization...")
        # Initialize segmenter
        segmenter = TextSegmenter(
            model_path="/scratch/project_2011770/bge-2048",
            prob_threshold=0.5,
            initial_min_words=15,
            max_groups=20,
        )

        # Quick test prediction
        print("\nTesting model prediction...")
        test_text = "This is a test sentence."
        probs, emb = segmenter.get_prediction(test_text)
        print("Test prediction successful!")
        print(f"Number of register probabilities: {len(probs)}")
    except Exception as e:
        print(f"Error during initialization: {str(e)}")
        raise

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
