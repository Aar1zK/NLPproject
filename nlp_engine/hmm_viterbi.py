"""
Algorithm 2: Hidden Markov Model (HMM) with Viterbi Dynamic Programming Decoding
Domain: Information Extraction (Sequence Labeling / Entity & Concept Extraction)

Mathematical Formulation:
1. Hidden State Space S = {PERSON, LOCATION, DATE, CONCEPT, OTHER}
2. Parameters (estimated via Maximum Likelihood with Laplace Smoothing + Morphological Backoff):
   - Initial State Probabilities: pi_k = P(y_1 = k)
   - Transition Matrix: A_{i,j} = P(y_t = j | y_{t-1} = i)
   - Emission Matrix: B_{k,w} = P(x_t = w | y_t = k) + Morphological Feature Prior P_morph(x_t | y_t = k)
3. Viterbi Decoding (Dynamic Programming in Log-Probability Space):
   - Initialization (t = 0):
     V[k, 0] = log(pi_k) + log(B_{k, x_0})
   - Recursion (t = 1 .. T-1):
     V[k, t] = max_{prev} ( V[prev, t-1] + log(A_{prev, k}) ) + log(B_{k, x_t})
     Backpointer[k, t] = argmax_{prev} ( V[prev, t-1] + log(A_{prev, k}) )
   - Termination & Backtracking:
     y^*_{T-1} = argmax_k V[k, T-1]
     y^*_{t-1} = Backpointer[y^*_t, t]
"""

import json
import math
import os
import re
from typing import List, Dict, Tuple, Any
import numpy as np
from .preprocessor import tokenize_with_punctuation, STOPWORDS


STATES = ["PERSON", "LOCATION", "DATE", "CONCEPT", "OTHER"]

PERSON_TITLES = {"dr", "dr.", "prof", "prof.", "mr", "mr.", "mrs", "mrs.", "ms", "ms.", "sir", "president", "emperor", "king", "queen", "general", "captain"}
LOCATION_PREPS = {"in", "at", "from", "to", "near", "across", "throughout", "inside", "outside"}
LOCATION_CUES = {
    "university", "institute", "laboratory", "college", "park", "city", "country",
    "republic", "kingdom", "empire", "ocean", "sea", "river", "mountain", "island",
    "peninsula", "continent", "england", "britain", "france", "germany", "italy",
    "spain", "russia", "china", "india", "japan", "america", "states", "united",
    "europe", "asia", "africa", "london", "paris", "berlin", "rome", "washington",
    "cambridge", "oxford", "princeton", "california", "utah", "hampshire", "new",
    "york", "boston", "geneva", "switzerland", "zurich", "vienna", "athens", "egypt",
    "bletchley", "dartmouth", "cornell", "aeronautical", "cern"
}
PERSON_KNOWN = {
    "alan", "turing", "albert", "einstein", "isaac", "newton", "charles", "darwin",
    "marie", "curie", "galileo", "galilei", "nikola", "tesla", "thomas", "edison",
    "ada", "lovelace", "babbage", "john", "mccarthy", "von", "neumann", "claude", "shannon",
    "noam", "chomsky", "frank", "rosenblatt", "geoffrey", "hinton", "yann", "lecun",
    "gregor", "mendel", "louis", "pasteur", "alexander", "fleming", "james", "watson",
    "francis", "crick", "rosalind", "franklin", "robert", "hooke", "anton", "leeuwenhoek",
    "napoleon", "bonaparte", "julius", "caesar", "abraham", "lincoln", "george",
    "winston", "churchill", "roosevelt", "mahatma", "gandhi", "nelson", "mandela",
    "aristotle", "plato", "socrates", "rene", "descartes", "immanuel", "kant", "karl", "marx",
    "adam", "smith", "maynard", "keynes", "tim", "berners-lee", "vint", "cerf", "bob", "kahn",
    "edsger", "dijkstra", "donald", "knuth", "dennis", "ritchie", "ken", "thompson", "linus", "torvalds"
}
SENTENCE_STARTERS_OTHER = {
    "however", "therefore", "moreover", "furthermore", "meanwhile", "nevertheless",
    "consequently", "additionally", "subsequently", "modern", "living", "during",
    "inside", "although", "because", "unlike", "each", "many", "several", "most"
}
CONCEPT_KNOWN = {
    "perceptron", "backpropagation", "enigma", "cipher", "algorithm", "intelligence",
    "network", "networks", "neural", "computability", "mitochondria", "ribosomes",
    "respiration", "homeostasis", "microscope", "double-helix", "prokaryotic",
    "eukaryotic", "cytoplasm", "membrane", "organelles", "packet", "packets",
    "protocol", "hypertext", "relativity", "gradient", "classification",
    "water", "surface", "layer", "level", "energy", "temperature", "pressure",
    "velocity", "density", "frequency", "wavelength", "amplitude", "voltage",
    "current", "resistance", "magnetic", "electric", "thermal", "chemical",
    "physical", "biological", "optical", "acoustic", "fluid", "liquid", "gas",
    "solid", "vapor", "atmosphere", "climate", "ocean", "wave", "waves", "flow",
    "flux", "radiation", "spectrum", "sensor", "signal", "image", "pixel",
    "matrix", "vector", "tensor", "dataset", "database", "feature", "features",
    "parameter", "parameters", "variable", "variables", "equation", "formula",
    "theorem", "lemma", "proof", "experiment", "measurement", "observation",
    "simulation", "accuracy", "precision", "error", "loss", "optimization",
    "distribution", "probability", "statistics", "regression", "clustering"
}
NON_PERSON_CAPITALIZED_WORDS = {
    "table", "figure", "fig", "section", "chapter", "abstract", "introduction",
    "conclusion", "summary", "results", "discussion", "method", "methods",
    "methodology", "references", "appendix", "equation", "step", "part", "type",
    "case", "study", "example", "note", "source", "data", "input", "output",
    "system", "model", "process", "analysis", "approach", "technique", "framework",
    "state", "phase", "stage", "mode", "rate", "ratio", "index", "factor", "value",
    "total", "average", "mean", "maximum", "minimum", "high", "low", "upper", "lower",
    "north", "south", "east", "west", "left", "right", "top", "bottom", "center"
}
HUMAN_ACTION_CONTEXT = {
    "published", "invented", "discovered", "proposed", "developed", "coined",
    "formulated", "designed", "introduced", "founded", "created", "wrote",
    "authored", "demonstrated", "proved", "worked", "born", "died", "argued",
    "stated", "concluded", "observed", "popularized", "described", "led"
}
MONTH_NAMES = {
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
    "jan.", "feb.", "mar.", "apr.", "aug.", "sept.", "oct.", "nov.", "dec.",
    "century", "bc", "bce", "ad", "ce"
}
CONCEPT_SUFFIXES = (
    "tion", "sion", "ism", "ity", "ment", "ness", "logy", "graphy", "metry",
    "nomy", "sis", "tomy", "pathy", "phobia", "philia", "cracy", "archy",
    "zyme", "some", "plast", "troph", "phage", "cyte", "kine", "ase",
    "algorithm", "theorem", "hypothesis", "protocol", "network", "matrix",
    "vector", "entropy", "model", "system", "equation", "function", "process",
    "mechanism", "structure", "principle", "law", "theory", "method", "architecture"
)


class HMMViterbiExtractor:
    def __init__(self, training_corpus_path: str = None, smoothing_k: float = 0.1):
        self.states = STATES
        self.state_to_idx = {s: i for i, s in enumerate(self.states)}
        self.num_states = len(self.states)
        self.smoothing_k = smoothing_k

        self.pi = np.zeros(self.num_states, dtype=float)
        self.transition = np.zeros((self.num_states, self.num_states), dtype=float)
        self.emission_counts: Dict[str, Dict[str, float]] = {s: {} for s in self.states}
        self.state_totals: Dict[str, float] = {s: 0.0 for s in self.states}
        self.vocab: set = set()

        self._train_from_corpus(training_corpus_path)

    def _train_from_corpus(self, corpus_path: str = None) -> None:
        """
        Estimates Initial (pi), Transition (A), and Lexical Emission (B) probability
        distributions via Maximum Likelihood Estimation (MLE) with Laplace smoothing.
        """
        if corpus_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            corpus_path = os.path.join(base_dir, "data", "hmm_training_corpus.json")

        training_sentences = []
        if os.path.exists(corpus_path):
            with open(corpus_path, "r", encoding="utf-8") as f:
                training_sentences = json.load(f)

        init_counts = np.full(self.num_states, self.smoothing_k, dtype=float)
        trans_counts = np.full((self.num_states, self.num_states), self.smoothing_k, dtype=float)

        for sent in training_sentences:
            tokens_tags = sent.get("tokens", [])
            if not tokens_tags:
                continue

            first_tag = tokens_tags[0][1]
            if first_tag in self.state_to_idx:
                init_counts[self.state_to_idx[first_tag]] += 1.0

            prev_idx = None
            for word, tag in tokens_tags:
                if tag not in self.state_to_idx:
                    tag = "OTHER"
                curr_idx = self.state_to_idx[tag]
                w_low = word.lower()
                self.vocab.add(w_low)
                self.emission_counts[tag][w_low] = self.emission_counts[tag].get(w_low, 0.0) + 1.0
                self.state_totals[tag] += 1.0

                if prev_idx is not None:
                    trans_counts[prev_idx, curr_idx] += 1.0
                prev_idx = curr_idx

        self.pi = init_counts / np.sum(init_counts)
        self.transition = trans_counts / np.sum(trans_counts, axis=1, keepdims=True)

    def compute_emission_prob(self, token: str, prev_token: str, next_token: str, pos_idx: int, state: str) -> float:
        """
        Computes P(token | state) combining lexical MLE probability with
        morphological/orthographic emission features so custom user-uploaded
        documents with unseen names, dates, locations, and technical concepts
        are accurately tagged without false-positive PERSON tags on CamelCase/headings.
        """
        w_low = token.lower()
        vocab_size = max(len(self.vocab), 100)

        # 1. Lexical smoothed probability from training corpus
        count = self.emission_counts[state].get(w_low, 0.0)
        total = self.state_totals[state]
        lexical_prob = (count + self.smoothing_k) / (total + self.smoothing_k * vocab_size)

        # 2. Morphological & Orthographic emission model for generalization
        is_punct = bool(re.match(r'^[.,!?;:()"\'\-]+$', token))
        is_stop = w_low in STOPWORDS
        is_four_digit_year = bool(re.match(r'^(1[0-9]{3}|20[0-9]{2})s?$', token))
        is_numeric = bool(re.match(r'^[0-9]+([.,:/\-][0-9]+)*(st|nd|rd|th|%|s)?$', w_low))
        is_month_or_era = w_low in MONTH_NAMES
        # Detect CamelCase / PascalCase compound technical identifiers (e.g., WaterSurface, NeuralNetwork)
        is_camel_case = bool(re.match(r'^[A-Za-z]+[a-z][A-Z][A-Za-z]*$', token))
        is_capitalized = token[0].isupper() and token[1:].islower() and len(token) > 1
        is_all_caps = token.isupper() and len(token) >= 2 and token.isalpha()
        is_hyphen_tech = "-" in token and not is_punct and len(token) > 4
        has_concept_suffix = w_low.endswith(CONCEPT_SUFFIXES) and len(w_low) >= 5
        # Check if compound word contains a known scientific/concept root (like 'water', 'surface', 'layer')
        contains_concept_root = any(
            root in w_low for root in ("water", "surface", "layer", "energy", "temp", "press", "wave", "flow", "data", "system", "model", "network", "cell", "graph", "matrix", "vector", "signal", "field")
        )

        prev_low = prev_token.lower() if prev_token else ""
        next_low = next_token.lower() if next_token else ""

        morph_score = 1e-4
        if is_punct or is_stop or (pos_idx == 0 and w_low in SENTENCE_STARTERS_OTHER) or w_low in NON_PERSON_CAPITALIZED_WORDS:
            if state == "OTHER":
                morph_score = 0.94
            else:
                morph_score = 0.0002
        elif is_four_digit_year or is_numeric or is_month_or_era:
            if state == "DATE":
                morph_score = 0.88
            elif state == "OTHER":
                morph_score = 0.05
            else:
                morph_score = 0.002
        elif is_camel_case or w_low in CONCEPT_KNOWN or contains_concept_root:
            if state == "CONCEPT":
                morph_score = 0.89
            elif state == "OTHER":
                morph_score = 0.09
            else:
                morph_score = 0.001
        elif w_low in PERSON_TITLES or w_low in PERSON_KNOWN:
            if state == "PERSON":
                morph_score = 0.88
            else:
                morph_score = 0.01
        elif w_low in LOCATION_CUES:
            if state == "LOCATION":
                morph_score = 0.86
            elif state == "CONCEPT" and w_low in {"university", "institute", "laboratory"}:
                morph_score = 0.08
            else:
                morph_score = 0.01
        elif is_capitalized:
            # Check contextual clues for Person vs Location vs Concept
            has_person_context = (
                prev_low.rstrip(".") in PERSON_TITLES
                or prev_low in PERSON_KNOWN
                or next_low in PERSON_KNOWN
                or next_low in HUMAN_ACTION_CONTEXT
                or prev_low in {"by", "researcher", "scientist", "professor", "author", "physicist", "biologist", "mathematician", "engineer"}
                or next_low == "et"
            )
            if prev_low.rstrip(".") in PERSON_TITLES or has_person_context:
                morph_score = 0.88 if state == "PERSON" else 0.02
            elif prev_low in LOCATION_PREPS:
                if state == "LOCATION":
                    morph_score = 0.78
                elif state == "CONCEPT":
                    morph_score = 0.12
                else:
                    morph_score = 0.03
            elif has_concept_suffix:
                morph_score = 0.78 if state == "CONCEPT" else (0.18 if state == "OTHER" else 0.01)
            elif pos_idx > 0:
                # Unseen capitalized word mid-sentence WITHOUT person context -> default to CONCEPT / OTHER, not PERSON
                if state == "CONCEPT":
                    morph_score = 0.48
                elif state == "OTHER":
                    morph_score = 0.30
                elif state == "LOCATION":
                    morph_score = 0.12
                elif state == "PERSON":
                    morph_score = 0.09
                else:
                    morph_score = 0.01
            else:
                # Sentence-initial capitalized word
                morph_score = 0.72 if state == "OTHER" else (0.18 if state == "CONCEPT" else 0.04)
        elif is_all_caps:
            # Acronyms like DNA, ATP, CPU, HTTP, NLP, RNA, SVD
            if state == "CONCEPT":
                morph_score = 0.80
            elif state == "LOCATION":
                morph_score = 0.12
            else:
                morph_score = 0.02
        elif has_concept_suffix or is_hyphen_tech:
            if state == "CONCEPT":
                morph_score = 0.72
            elif state == "OTHER":
                morph_score = 0.24
            else:
                morph_score = 0.01
        else:
            # Regular lowercase content word
            if state == "OTHER":
                morph_score = 0.76
            elif state == "CONCEPT" and len(w_low) >= 7:
                morph_score = 0.20
            else:
                morph_score = 0.005

        # Interpolate lexical MLE probability (when word is known in corpus) with morphological model
        if count > 0:
            return 0.65 * lexical_prob + 0.35 * morph_score
        return 0.15 * lexical_prob + 0.85 * morph_score

    def viterbi_decode(self, tokens: List[str]) -> Tuple[List[str], np.ndarray, np.ndarray]:
        """
        Executes the Viterbi Dynamic Programming algorithm over a token sequence.
        Returns:
          - best_tags: Optimal sequence of states [y_0, ..., y_{T-1}]
          - trellis: Log-probability matrix V[k, t] of shape (K, T)
          - backpointers: Backpointer matrix B[k, t] of shape (K, T)
        """
        T = len(tokens)
        K = self.num_states
        if T == 0:
            return [], np.zeros((K, 0)), np.zeros((K, 0), dtype=int)

        trellis = np.full((K, T), -1e9, dtype=float)
        backpointers = np.zeros((K, T), dtype=int)

        log_pi = np.log(np.clip(self.pi, 1e-12, 1.0))
        log_A = np.log(np.clip(self.transition, 1e-12, 1.0))

        # Step 1: Initialization (t = 0)
        next_tok_0 = tokens[1] if T > 1 else ""
        for k, state in enumerate(self.states):
            em_prob = self.compute_emission_prob(tokens[0], "", next_tok_0, 0, state)
            trellis[k, 0] = log_pi[k] + math.log(max(em_prob, 1e-12))
            backpointers[k, 0] = 0

        # Step 2: Dynamic Programming Recursion (t = 1 .. T-1)
        for t in range(1, T):
            # Look back past a single period if previous token was an abbreviation dot (e.g. "Dr" "." "Sarah")
            prev_tok = tokens[t - 2] if (tokens[t - 1] == "." and t >= 2) else tokens[t - 1]
            curr_tok = tokens[t]
            next_tok = tokens[t + 1] if t + 1 < T else ""
            for k, state in enumerate(self.states):
                em_prob = self.compute_emission_prob(curr_tok, prev_tok, next_tok, t, state)
                log_em = math.log(max(em_prob, 1e-12))

                candidate_scores = trellis[:, t - 1] + log_A[:, k]
                best_prev = int(np.argmax(candidate_scores))
                trellis[k, t] = candidate_scores[best_prev] + log_em
                backpointers[k, t] = best_prev

        # Step 3: Termination & Backtracking
        best_path_indices = [0] * T
        best_path_indices[T - 1] = int(np.argmax(trellis[:, T - 1]))
        for t in range(T - 1, 0, -1):
            best_path_indices[t - 1] = int(backpointers[best_path_indices[t], t])

        best_tags = [self.states[idx] for idx in best_path_indices]
        return best_tags, trellis, backpointers

    def extract_entities_and_flashcards(self, sentences: List[str]) -> Dict[str, Any]:
        """
        Runs HMM + Viterbi decoding across all sentences in the document to:
        1. Group contiguous tagged tokens into structured multi-word entities
           (PERSON, LOCATION, DATE, CONCEPT).
        2. Generate Study Flashcards linking extracted entities/concepts to their context sentences.
        3. Capture the Viterbi Trellis table on a representative sentence for the Math Inspector.
        """
        entities_by_type: Dict[str, List[Dict[str, Any]]] = {
            "PERSON": [],
            "LOCATION": [],
            "DATE": [],
            "CONCEPT": []
        }
        seen_normalized: Dict[str, set] = {k: set() for k in entities_by_type}
        sentence_annotations: List[Dict[str, Any]] = []

        sample_trellis_data = None

        for sent_idx, sent in enumerate(sentences):
            tokens = tokenize_with_punctuation(sent)
            if not tokens:
                continue

            tags, trellis, backpointers = self.viterbi_decode(tokens)

            # Save a representative Viterbi trellis (first sentence with >= 6 tokens and at least 1 entity)
            if sample_trellis_data is None and len(tokens) >= 6 and any(t != "OTHER" for t in tags):
                preview_len = min(len(tokens), 12)
                sample_trellis_data = {
                    "sentence_index": sent_idx,
                    "sentence": sent,
                    "tokens": tokens[:preview_len],
                    "decoded_tags": tags[:preview_len],
                    "states": self.states,
                    "trellis_log_probs": np.round(trellis[:, :preview_len], 2).tolist(),
                    "backpointers": backpointers[:, :preview_len].tolist()
                }

            # Merge contiguous spans of identical non-OTHER tags
            sent_spans = []
            i = 0
            while i < len(tokens):
                tag = tags[i]
                if tag != "OTHER":
                    span_tokens = [tokens[i]]
                    j = i + 1
                    while j < len(tokens) and tags[j] == tag:
                        span_tokens.append(tokens[j])
                        j += 1
                    phrase = " ".join(span_tokens).strip(" .,;:()\"'")
                    if len(phrase) >= 2 and phrase.lower() not in STOPWORDS:
                        sent_spans.append({
                            "text": phrase,
                            "label": tag,
                            "start_token": i,
                            "end_token": j - 1
                        })
                        norm_key = phrase.lower()
                        if norm_key not in seen_normalized[tag]:
                            seen_normalized[tag].add(norm_key)
                            entities_by_type[tag].append({
                                "entity": phrase,
                                "type": tag,
                                "sentence_index": sent_idx,
                                "context_sentence": sent
                            })
                    i = j
                else:
                    i += 1

            sentence_annotations.append({
                "sentence_index": sent_idx,
                "tokens": tokens,
                "tags": tags,
                "spans": sent_spans
            })

        # Build Study Flashcards from extracted entities & concepts
        flashcards = self._build_study_flashcards(entities_by_type, sentences)

        return {
            "entities_by_type": entities_by_type,
            "flashcards": flashcards,
            "sentence_annotations": sentence_annotations,
            "math_details": {
                "algorithm": "Hidden Markov Model (HMM) + Viterbi Dynamic Programming",
                "hidden_states": self.states,
                "initial_probabilities": [round(float(x), 4) for x in self.pi],
                "transition_matrix": np.round(self.transition, 4).tolist(),
                "sample_viterbi_trellis": sample_trellis_data
            }
        }

    def _build_study_flashcards(
        self,
        entities_by_type: Dict[str, List[Dict[str, Any]]],
        sentences: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Creates interactive Q&A study flashcards from the HMM-extracted entities and concepts.
        """
        cards: List[Dict[str, Any]] = []

        # Prioritize key people, concepts, dates, and locations
        for item in entities_by_type["PERSON"][:4]:
            cards.append({
                "category": "Key Figure (PERSON)",
                "front": f"Who is {item['entity']} and what role did they play in this text?",
                "back": item["context_sentence"],
                "entity": item["entity"],
                "sentence_index": item["sentence_index"]
            })

        for item in entities_by_type["CONCEPT"][:5]:
            cards.append({
                "category": "Core Concept (CONCEPT)",
                "front": f"What is the significance of '{item['entity']}'?",
                "back": item["context_sentence"],
                "entity": item["entity"],
                "sentence_index": item["sentence_index"]
            })

        for item in entities_by_type["DATE"][:3]:
            cards.append({
                "category": "Historical Date / Metric (DATE)",
                "front": f"What happened in or is associated with '{item['entity']}'?",
                "back": item["context_sentence"],
                "entity": item["entity"],
                "sentence_index": item["sentence_index"]
            })

        for item in entities_by_type["LOCATION"][:3]:
            cards.append({
                "category": "Location / Institution (LOCATION)",
                "front": f"How is {item['entity']} relevant to the topic?",
                "back": item["context_sentence"],
                "entity": item["entity"],
                "sentence_index": item["sentence_index"]
            })

        return cards[:10]
