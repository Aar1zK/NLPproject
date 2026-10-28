"""
Algorithm 4: Multinomial Naive Bayes Question Classifier + Simplified Lesk WSD & Answer Pinpointing
Domain: Question & Answering (Q&A)

Mathematical Formulation:
Part A - Expected Answer Type (EAT) Classification via Multinomial Naive Bayes:
  Given a question Q = (w_1, w_2, ..., w_m) with bigrams/unigrams F(Q):
    c^* = argmax_{c in C} [ log P(c) + sum_{f in F(Q)} log P(f | c) ]
  where C = {PERSON, LOCATION, DATE, CONCEPT} and conditional probabilities use Laplace smoothing:
    P(f | c) = (count(f, c) + alpha) / (sum_{f'} count(f', c) + alpha * |V_Q|)

Part B - Candidate Answer Span Pinpointing (Proximity & Type Alignment Scoring):
  Given the top retrieved passage S from BM25 and HMM-Viterbi entity spans E(S):
    Score(span_j) = w_type * I(tag(span_j) == c^*) + w_prox * (1 / (1 + min_dist(span_j, Q_terms)))
  Extracts the exact concise answer phrase (1-6 words) + supporting sentence.

Part C - Simplified Lesk Algorithm for Word Sense Disambiguation (WSD):
  For ambiguous content terms w in the question/answer:
    sense^* = argmax_{s in Senses(w)} | Context(w) intersect Signature(s) |
  where Context(w) is the bag of stemmed words in the passage + question, and
  Signature(s) is the bag of stemmed words in dictionary sense s's gloss + examples.
"""

import json
import math
import os
import re
from typing import List, Dict, Any, Tuple
from .preprocessor import tokenize_words, preprocess_sentence, stem_word, STOPWORDS


ANSWER_TYPES = ["PERSON", "LOCATION", "DATE", "CONCEPT"]


class NaiveBayesLeskQA:
    def __init__(self, qa_train_path: str = None, synset_path: str = None, alpha: float = 0.5):
        self.classes = ANSWER_TYPES
        self.alpha = alpha
        self.class_priors: Dict[str, float] = {}
        self.feature_counts: Dict[str, Dict[str, float]] = {c: {} for c in self.classes}
        self.class_totals: Dict[str, float] = {c: 0.0 for c in self.classes}
        self.vocab: set = set()
        self.synset_dict: Dict[str, List[Dict[str, Any]]] = {}

        self._train_naive_bayes(qa_train_path)
        self._load_synsets(synset_path)

    def _extract_question_features(self, question: str) -> List[str]:
        """
        Extracts lexical unigrams, question-word prefix features, and bigrams
        from a question for Multinomial Naive Bayes classification.
        """
        tokens = tokenize_words(question, lowercase=True)
        if not tokens:
            return []

        features = []
        # Unigrams (stemmed content & wh-words)
        for i, tok in enumerate(tokens):
            features.append(f"uni:{tok}")
            if i == 0:
                features.append(f"first:{tok}")
            if i < len(tokens) - 1:
                features.append(f"bi:{tok}_{tokens[i+1]}")
        return features

    def _train_naive_bayes(self, qa_train_path: str = None) -> None:
        """
        Trains the Multinomial Naive Bayes question classifier from labeled questions.
        """
        if qa_train_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            qa_train_path = os.path.join(base_dir, "data", "qa_training_data.json")

        dataset = []
        if os.path.exists(qa_train_path):
            with open(qa_train_path, "r", encoding="utf-8") as f:
                dataset = json.load(f)

        class_doc_counts = {c: 0 for c in self.classes}
        total_docs = max(len(dataset), 1)

        for item in dataset:
            q_text = item.get("question", "")
            label = item.get("answer_type", "CONCEPT")
            if label not in self.classes:
                label = "CONCEPT"

            class_doc_counts[label] += 1
            feats = self._extract_question_features(q_text)
            for feat in feats:
                self.vocab.add(feat)
                self.feature_counts[label][feat] = self.feature_counts[label].get(feat, 0.0) + 1.0
                self.class_totals[label] += 1.0

        for c in self.classes:
            self.class_priors[c] = (class_doc_counts[c] + 1.0) / (total_docs + len(self.classes))

    def _load_synsets(self, synset_path: str = None) -> None:
        """
        Loads polysemous word sense definitions (glosses & examples) for the Lesk algorithm.
        """
        if synset_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            synset_path = os.path.join(base_dir, "data", "synset_dictionary.json")

        if os.path.exists(synset_path):
            with open(synset_path, "r", encoding="utf-8") as f:
                self.synset_dict = json.load(f)

    def classify_question(self, question: str) -> Dict[str, Any]:
        """
        Computes posterior probabilities P(c | Q) across all Expected Answer Types
        using Multinomial Naive Bayes in log-space and softmax normalization.
        """
        features = self._extract_question_features(question)
        vocab_size = max(len(self.vocab), 50)

        log_scores: Dict[str, float] = {}
        feature_likelihoods: Dict[str, List[Dict[str, Any]]] = {c: [] for c in self.classes}

        for c in self.classes:
            log_prob = math.log(self.class_priors.get(c, 0.25))
            denom = self.class_totals[c] + self.alpha * vocab_size
            for feat in features:
                count = self.feature_counts[c].get(feat, 0.0)
                cond_prob = (count + self.alpha) / denom
                log_prob += math.log(cond_prob)
                feature_likelihoods[c].append({
                    "feature": feat,
                    "count_in_class": int(count),
                    "cond_prob": round(cond_prob, 5)
                })
            log_scores[c] = log_prob

        # Convert log-scores to normalized posterior probabilities via log-sum-exp trick
        max_log = max(log_scores.values())
        exp_scores = {c: math.exp(val - max_log) for c, val in log_scores.items()}
        sum_exp = sum(exp_scores.values())
        posteriors = {c: round(exp_scores[c] / sum_exp, 4) for c in self.classes}

        predicted_class = max(posteriors.items(), key=lambda x: x[1])[0]

        return {
            "predicted_answer_type": predicted_class,
            "posterior_probabilities": posteriors,
            "log_probabilities": {c: round(val, 4) for c, val in log_scores.items()},
            "extracted_features": features,
            "feature_likelihoods_top_class": feature_likelihoods[predicted_class][:8]
        }

    def pinpoint_answer(
        self,
        question: str,
        expected_type: str,
        top_passages: List[Dict[str, Any]],
        sentence_annotations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Combines BM25 top passages + HMM-Viterbi entity spans + keyword distance scoring
        to extract the exact concise answer phrase and supporting evidence sentence.
        """
        if not top_passages:
            return {
                "exact_answer": "No relevant passage found in the document.",
                "supporting_sentence": "",
                "sentence_index": -1,
                "candidate_spans_scored": []
            }

        q_stemmed = set(preprocess_sentence(question, remove_stopwords=True, apply_stemming=True))
        q_raw_lower = set(tokenize_words(question, lowercase=True))

        # Map sentence_index -> annotation
        ann_map = {ann["sentence_index"]: ann for ann in sentence_annotations}

        candidate_spans_scored: List[Dict[str, Any]] = []

        # Inspect top 3 BM25 passages
        for rank_idx, passage in enumerate(top_passages[:3]):
            sent_idx = passage["sentence_index"]
            sent_text = passage["text"]
            bm25_score = passage["bm25_score"]
            ann = ann_map.get(sent_idx, {"tokens": [], "tags": [], "spans": []})
            tokens = ann.get("tokens", tokenize_words(sent_text, lowercase=False))

            # Locate indices of query terms inside this sentence
            query_positions = [
                i for i, tok in enumerate(tokens)
                if stem_word(tok.lower()) in q_stemmed
            ]

            for span in ann.get("spans", []):
                span_text = span["text"]
                span_label = span["label"]
                span_words_lower = set(tokenize_words(span_text, lowercase=True))
                span_stems = {stem_word(w) for w in span_words_lower}

                # Skip candidate spans that are already verbatim in the question (unless it's a CONCEPT definition query)
                overlap_with_question = len(span_stems.intersection(q_stemmed))
                is_wholly_in_question = span_words_lower.issubset(q_raw_lower)
                if is_wholly_in_question and expected_type != "CONCEPT":
                    continue

                # 1. Type alignment score: Does HMM tag match Naive Bayes Expected Answer Type?
                if span_label == expected_type:
                    type_bonus = 3.0
                elif expected_type == "CONCEPT" and span_label in ("PERSON", "LOCATION", "DATE"):
                    type_bonus = 0.8
                else:
                    type_bonus = 0.4

                # 2. Keyword proximity score: Inverse token distance to matched question words
                if query_positions:
                    span_center = 0.5 * (span["start_token"] + span["end_token"])
                    min_dist = min(abs(span_center - qp) for qp in query_positions)
                    proximity_score = 1.0 / (1.0 + 0.25 * min_dist)
                else:
                    proximity_score = 0.3

                # 3. Novelty bonus (prefer spans not already repeated in the question)
                novelty_bonus = 1.0 if overlap_with_question == 0 else 0.4

                # 4. Combine with passage BM25 weight
                passage_weight = max(bm25_score, 0.2) / (1.0 + 0.35 * rank_idx)
                final_span_score = passage_weight * (type_bonus + 1.5 * proximity_score) * novelty_bonus

                candidate_spans_scored.append({
                    "span_text": span_text,
                    "hmm_tag": span_label,
                    "expected_type_match": span_label == expected_type,
                    "sentence_index": sent_idx,
                    "bm25_passage_score": round(bm25_score, 4),
                    "proximity_score": round(proximity_score, 4),
                    "final_score": round(final_span_score, 4),
                    "supporting_sentence": sent_text
                })

        candidate_spans_scored.sort(key=lambda x: x["final_score"], reverse=True)

        best_passage = top_passages[0]
        best_sent_text = best_passage["text"]
        best_sent_idx = best_passage["sentence_index"]

        # For explanatory/definition questions ("What is...", "Why...", "How..."), check if a definitional clause exists
        q_lower = question.strip().lower()
        is_explanatory = q_lower.startswith(("what is", "what are", "how does", "how do", "why ", "explain ", "define "))

        if candidate_spans_scored and not is_explanatory:
            top_cand = candidate_spans_scored[0]
            exact_answer = top_cand["span_text"]
            best_sent_text = top_cand["supporting_sentence"]
            best_sent_idx = top_cand["sentence_index"]
        else:
            # Extract concise clause or predicate phrase around matched keywords
            exact_answer = self._extract_concise_clause(best_sent_text, q_stemmed)

        return {
            "exact_answer": exact_answer,
            "supporting_sentence": best_sent_text,
            "sentence_index": best_sent_idx,
            "candidate_spans_scored": candidate_spans_scored[:6]
        }

    def _extract_concise_clause(self, sentence: str, q_stemmed: set) -> str:
        """
        Extracts a focused definitional or predicate clause when a question asks
        for a description/definition ("What is X?", "How does X work?").
        """
        # Look for copula/definitional patterns ("is a ...", "refers to ...", "is defined as ...")
        patterns = [
            r"\b(?:is defined as|refers to|is known as|consists of|is a|are a|is the|was a|was the)\s+([^.;]+)",
            r"\b(?:because|by|through|using|via)\s+([^.;]+)"
        ]
        for pat in patterns:
            m = re.search(pat, sentence, flags=re.IGNORECASE)
            if m and len(m.group(0).split()) >= 4:
                return m.group(0).strip(" ,.;:")

        return sentence

    def run_lesk_wsd(self, question: str, supporting_sentence: str, all_sentences: List[str]) -> List[Dict[str, Any]]:
        """
        Executes the Simplified Lesk Algorithm to disambiguate polysemous terms
        appearing in the question or supporting answer passage:
          sense^* = argmax_{s in Senses(w)} | Context(w) intersect Signature(s) |
        Also supports dynamic contextual sense synthesis for custom user documents!
        """
        combined_text = f"{question} {supporting_sentence}"
        raw_tokens = tokenize_words(combined_text, lowercase=True)
        context_stems = set(preprocess_sentence(combined_text, remove_stopwords=True, apply_stemming=True))

        # Include stems from neighboring sentences for richer context window
        for s in all_sentences[:5]:
            if supporting_sentence and s == supporting_sentence:
                continue
        disambiguated_results: List[Dict[str, Any]] = []
        seen_words = set()

        # 1. Check tokens against our multi-sense dictionary (exact word or stem match)
        for tok in raw_tokens:
            if tok in STOPWORDS or len(tok) <= 2:
                continue
            tok_stem = stem_word(tok)
            matched_key = None
            if tok in self.synset_dict:
                matched_key = tok
            else:
                for dict_word in self.synset_dict:
                    if stem_word(dict_word) == tok_stem:
                        matched_key = dict_word
                        break

            if matched_key and matched_key not in seen_words:
                seen_words.add(matched_key)
                senses = self.synset_dict[matched_key]
                sense_evaluations = []

                for s in senses:
                    sig_text = f"{s['sense_id']} {s['domain']} {s['definition']} {s.get('example', '')} {' '.join(s.get('keywords', []))}"
                    sig_stems = set(preprocess_sentence(sig_text, remove_stopwords=True, apply_stemming=True))
                    # Exclude the target word itself from overlap
                    sig_stems.discard(tok_stem)
                    sig_stems.discard(stem_word(matched_key))

                    overlap_terms = sorted(list(context_stems.intersection(sig_stems)))
                    overlap_count = len(overlap_terms)

                    sense_evaluations.append({
                        "sense_id": s["sense_id"],
                        "domain": s["domain"],
                        "definition": s["definition"],
                        "example": s.get("example", ""),
                        "overlap_count": overlap_count,
                        "overlapping_stems": overlap_terms
                    })

                sense_evaluations.sort(key=lambda x: x["overlap_count"], reverse=True)
                best_sense = sense_evaluations[0]

                disambiguated_results.append({
                    "word": matched_key,
                    "selected_sense": best_sense["sense_id"],
                    "domain": best_sense["domain"],
                    "definition": best_sense["definition"],
                    "overlap_score": best_sense["overlap_count"],
                    "overlapping_stems": best_sense["overlapping_stems"],
                    "candidate_senses": sense_evaluations
                })

        # 2. If user uploaded a custom text whose words aren't in the static synset table,
        # dynamically construct a Lesk contextual vs. general sense comparison for the top content word
        if not disambiguated_results and supporting_sentence:
            content_words = [
                w for w in tokenize_words(supporting_sentence, lowercase=True)
                if w not in STOPWORDS and len(w) >= 5 and w.isalpha()
            ]
            if content_words:
                target_word = content_words[0]
                target_stem = stem_word(target_word)
                # Gather sentences in the document that mention this stem vs. sentences that don't
                matching_sents = [s for s in all_sentences if target_stem in preprocess_sentence(s)]
                doc_sig_text = " ".join(matching_sents[:2])
                doc_sig_stems = set(preprocess_sentence(doc_sig_text)) - {target_stem}
                overlap_terms = sorted(list(context_stems.intersection(doc_sig_stems)))[:6]

                candidates = [
                    {
                        "sense_id": f"{target_word}.n.01 (Document-Specific Technical Sense)",
                        "domain": "Document Context",
                        "definition": f"Contextual meaning in passage: '{supporting_sentence[:120]}...'",
                        "overlap_count": len(overlap_terms),
                        "overlapping_stems": overlap_terms
                    },
                    {
                        "sense_id": f"{target_word}.n.02 (General Out-of-Context Sense)",
                        "domain": "General Lexicon",
                        "definition": f"Generic everyday usage of '{target_word}' unrelated to the passage domain.",
                        "overlap_count": 0,
                        "overlapping_stems": []
                    }
                ]
                disambiguated_results.append({
                    "word": target_word,
                    "selected_sense": candidates[0]["sense_id"],
                    "domain": candidates[0]["domain"],
                    "definition": candidates[0]["definition"],
                    "overlap_score": candidates[0]["overlap_count"],
                    "overlapping_stems": candidates[0]["overlapping_stems"],
                    "candidate_senses": candidates
                })

        return disambiguated_results

    def answer_question(
        self,
        question: str,
        bm25_result: Dict[str, Any],
        sentence_annotations: List[Dict[str, Any]],
        all_sentences: List[str]
    ) -> Dict[str, Any]:
        """
        Full Algorithm 4 pipeline:
        1. Classify Expected Answer Type (PERSON, LOCATION, DATE, CONCEPT) via Multinomial Naive Bayes.
        2. Pinpoint the exact answer span inside the top BM25 passage using HMM tags & keyword proximity.
        3. Disambiguate key polysemous terms in the question/answer using the Lesk Algorithm.
        """
        nb_classification = self.classify_question(question)
        expected_type = nb_classification["predicted_answer_type"]

        top_passages = bm25_result.get("top_passages", [])
        pinpointed = self.pinpoint_answer(
            question=question,
            expected_type=expected_type,
            top_passages=top_passages,
            sentence_annotations=sentence_annotations
        )

        lesk_results = self.run_lesk_wsd(
            question=question,
            supporting_sentence=pinpointed["supporting_sentence"],
            all_sentences=all_sentences
        )

        return {
            "question": question,
            "expected_answer_type": expected_type,
            "exact_answer": pinpointed["exact_answer"],
            "supporting_sentence": pinpointed["supporting_sentence"],
            "sentence_index": pinpointed["sentence_index"],
            "lesk_disambiguations": lesk_results,
            "math_details": {
                "algorithm": "Multinomial Naive Bayes EAT Classifier + Simplified Lesk WSD + Span Proximity Pinpointer",
                "naive_bayes": nb_classification,
                "candidate_spans_scored": pinpointed["candidate_spans_scored"],
                "lesk_wsd_evaluations": lesk_results
            }
        }
