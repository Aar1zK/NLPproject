"""
Algorithm 3: Okapi BM25 Probabilistic Ranking Model
Domain: Information Retrieval (Passage / Evidence Sentence Retrieval)

Mathematical Formulation:
Given a user query Q = {q_1, q_2, ..., q_m} and a candidate sentence/passage D_i:
  BM25(Q, D_i) = sum_{q in Q} IDF(q) * [ (f(q, D_i) * (k_1 + 1)) / (f(q, D_i) + k_1 * (1 - b + b * (|D_i| / avgdl))) ]
where:
  - f(q, D_i) = term frequency of stemmed query term q in passage D_i
  - |D_i| = length (number of content tokens) of passage D_i
  - avgdl = average passage length across the entire document N
  - k_1 = term frequency saturation hyperparameter (default: 1.5)
  - b = document length normalization hyperparameter (default: 0.75)
  - IDF(q) = log( 1 + (N - n(q) + 0.5) / (n(q) + 0.5) )
    where n(q) is the number of passages containing term q.
"""

import math
from typing import List, Dict, Any
from .preprocessor import preprocess_sentence, tokenize_words


class OkapiBM25Retriever:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.sentences: List[str] = []
        self.tokenized_sentences: List[List[str]] = []
        self.raw_tokens_sentences: List[List[str]] = []
        self.doc_freqs: Dict[str, int] = {}
        self.idf_scores: Dict[str, float] = {}
        self.doc_lengths: List[int] = []
        self.avgdl: float = 1.0
        self.N: int = 0

    def index_sentences(self, sentences: List[str]) -> None:
        """
        Builds an inverted index and precomputes passage lengths and BM25 IDF weights.
        """
        self.sentences = sentences
        self.N = len(sentences)
        self.tokenized_sentences = [
            preprocess_sentence(s, remove_stopwords=True, apply_stemming=True)
            for s in sentences
        ]
        self.raw_tokens_sentences = [
            tokenize_words(s, lowercase=True)
            for s in sentences
        ]
        self.doc_lengths = [len(toks) for toks in self.tokenized_sentences]
        self.avgdl = (sum(self.doc_lengths) / float(self.N)) if self.N > 0 else 1.0
        if self.avgdl == 0:
            self.avgdl = 1.0

        self.doc_freqs = {}
        for toks in self.tokenized_sentences:
            for unique_term in set(toks):
                self.doc_freqs[unique_term] = self.doc_freqs.get(unique_term, 0) + 1

        self.idf_scores = {}
        for term, df in self.doc_freqs.items():
            # Robertson-Sparck Jones smooth non-negative IDF formula
            self.idf_scores[term] = math.log(1.0 + (self.N - df + 0.5) / (df + 0.5))

    def retrieve(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Scores every sentence in the indexed document against the query using Okapi BM25
        and returns the top-k passages along with a full mathematical breakdown.
        """
        if self.N == 0:
            return {
                "query": query,
                "query_terms": [],
                "top_passages": [],
                "math_details": {}
            }

        query_terms = preprocess_sentence(query, remove_stopwords=True, apply_stemming=True)
        # Fallback if query only contained stopwords (e.g. "What is it?")
        if not query_terms:
            query_terms = preprocess_sentence(query, remove_stopwords=False, apply_stemming=True)

        query_term_weights: List[Dict[str, Any]] = []
        for q in query_terms:
            df = self.doc_freqs.get(q, 0)
            idf = self.idf_scores.get(q, math.log(1.0 + (self.N + 0.5) / 0.5))
            query_term_weights.append({
                "stemmed_term": q,
                "doc_frequency": df,
                "idf": round(idf, 4)
            })

        scored_passages: List[Dict[str, Any]] = []

        for idx in range(self.N):
            sent_tokens = self.tokenized_sentences[idx]
            dl = self.doc_lengths[idx]
            length_norm = 1.0 - self.b + self.b * (dl / self.avgdl)

            total_score = 0.0
            term_contributions: List[Dict[str, Any]] = []

            for q in query_terms:
                tf = sent_tokens.count(q)
                df = self.doc_freqs.get(q, 0)
                idf = self.idf_scores.get(q, 0.0)

                if tf > 0:
                    numerator = tf * (self.k1 + 1.0)
                    denominator = tf + self.k1 * length_norm
                    tf_sat = numerator / denominator
                    term_score = idf * tf_sat
                else:
                    tf_sat = 0.0
                    term_score = 0.0

                total_score += term_score
                term_contributions.append({
                    "term": q,
                    "tf": tf,
                    "idf": round(idf, 4),
                    "tf_saturation": round(tf_sat, 4),
                    "term_bm25": round(term_score, 4)
                })

            scored_passages.append({
                "sentence_index": idx,
                "text": self.sentences[idx],
                "bm25_score": round(total_score, 4),
                "doc_length": dl,
                "length_norm_factor": round(length_norm, 4),
                "term_contributions": term_contributions
            })

        # Sort descending by BM25 score
        scored_passages.sort(key=lambda x: x["bm25_score"], reverse=True)
        top_passages = scored_passages[:top_k]

        return {
            "query": query,
            "query_terms": query_terms,
            "top_passages": top_passages,
            "all_scores": [
                {
                    "sentence_index": p["sentence_index"],
                    "bm25_score": p["bm25_score"]
                }
                for p in scored_passages[:10]
            ],
            "math_details": {
                "algorithm": "Okapi BM25 Probabilistic Retrieval Model",
                "k1": self.k1,
                "b": self.b,
                "total_passages_N": self.N,
                "average_passage_length_avgdl": round(self.avgdl, 2),
                "query_term_idf_table": query_term_weights,
                "top_passage_breakdown": top_passages[0] if top_passages else None
            }
        }
