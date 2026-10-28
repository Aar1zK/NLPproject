"""
Algorithm 1: TextRank (Graph-Based Extractive Text Summarization)
Domain: Text Summarization

Mathematical Formulation:
1. Represent each sentence S_i as a TF-IDF vector in R^|V|:
   tf(t, S_i) = count(t in S_i) / |S_i|
   idf(t) = log((N + 1) / (df(t) + 1)) + 1
2. Construct a weighted undirected graph G = (V, E) where vertices V are sentences
   and edge weights W_{i,j} are Cosine Similarities:
   W_{i,j} = (v_i . v_j) / (||v_i||_2 * ||v_j||_2)
3. Row-normalize W into a Markov Transition Matrix M:
   M_{i,j} = W_{i,j} / sum_k W_{i,k}
4. Solve for the stationary PageRank centrality vector p via Power Iteration:
   p^{(t+1)} = (1 - d) / N * 1 + d * M^T * p^{(t)}
   until ||p^{(t+1)} - p^{(t)}||_1 < epsilon.
"""

import math
from typing import List, Dict, Any, Tuple
import numpy as np
from .preprocessor import preprocess_sentence


class TextRankSummarizer:
    def __init__(self, damping_factor: float = 0.85, max_iter: int = 100, tol: float = 1e-6):
        self.damping_factor = damping_factor
        self.max_iter = max_iter
        self.tol = tol

    def build_tfidf_matrix(self, tokenized_sentences: List[List[str]]) -> Tuple[np.ndarray, List[str]]:
        """
        Builds a Sentence-by-Term TF-IDF matrix of shape (N_sentences, |V|).
        """
        n_sents = len(tokenized_sentences)
        vocab = sorted(list({term for sent in tokenized_sentences for term in sent}))
        vocab_index = {term: idx for idx, term in enumerate(vocab)}
        n_terms = len(vocab)

        if n_sents == 0 or n_terms == 0:
            return np.zeros((n_sents, 0)), []

        # Document frequency (number of sentences containing term t)
        df = np.zeros(n_terms, dtype=float)
        tf_matrix = np.zeros((n_sents, n_terms), dtype=float)

        for i, sent in enumerate(tokenized_sentences):
            if not sent:
                continue
            seen_in_sent = set()
            for term in sent:
                idx = vocab_index[term]
                tf_matrix[i, idx] += 1.0
                seen_in_sent.add(idx)
            # Normalize TF by sentence length
            tf_matrix[i, :] /= float(len(sent))
            for idx in seen_in_sent:
                df[idx] += 1.0

        # Smooth IDF: log((N + 1) / (df + 1)) + 1
        idf = np.log((n_sents + 1.0) / (df + 1.0)) + 1.0
        tfidf_matrix = tf_matrix * idf
        return tfidf_matrix, vocab

    def compute_similarity_matrix(self, tfidf_matrix: np.ndarray) -> np.ndarray:
        """
        Computes pairwise Cosine Similarity matrix W (N x N) with zero self-loops.
        """
        n_sents = tfidf_matrix.shape[0]
        if n_sents == 0:
            return np.zeros((0, 0))

        norms = np.linalg.norm(tfidf_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        normalized = tfidf_matrix / norms
        sim_matrix = np.dot(normalized, normalized.T)

        # Remove self-loops (diagonal = 0)
        np.fill_diagonal(sim_matrix, 0.0)
        return np.clip(sim_matrix, 0.0, 1.0)

    def build_transition_matrix(self, sim_matrix: np.ndarray) -> np.ndarray:
        """
        Row-normalizes the similarity adjacency matrix into a stochastic Markov matrix M.
        Dangling nodes (rows with sum 0) receive a uniform distribution 1/N.
        """
        n = sim_matrix.shape[0]
        if n == 0:
            return np.zeros((0, 0))

        transition = np.zeros_like(sim_matrix, dtype=float)
        row_sums = sim_matrix.sum(axis=1)
        for i in range(n):
            if row_sums[i] > 1e-10:
                transition[i, :] = sim_matrix[i, :] / row_sums[i]
            else:
                transition[i, :] = 1.0 / float(n)
        return transition

    def power_iteration(self, transition_matrix: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Numerically solves for the PageRank eigenvector via Power Iteration:
        p^{(k+1)} = (1 - d)/N + d * M^T * p^{(k)}
        """
        n = transition_matrix.shape[0]
        if n == 0:
            return np.array([]), []

        p = np.ones(n, dtype=float) / float(n)
        teleport = (1.0 - self.damping_factor) / float(n)
        history = [{
            "iteration": 0,
            "l1_residual": 1.0,
            "scores": [round(float(x), 5) for x in p]
        }]

        for it in range(1, self.max_iter + 1):
            p_next = teleport + self.damping_factor * np.dot(transition_matrix.T, p)
            # Normalize to ensure sum(p) = 1.0
            p_next = p_next / np.sum(p_next)
            residual = float(np.sum(np.abs(p_next - p)))
            p = p_next

            if it <= 6 or residual < self.tol or it == self.max_iter:
                history.append({
                    "iteration": it,
                    "l1_residual": round(residual, 8),
                    "scores": [round(float(x), 5) for x in p]
                })

            if residual < self.tol:
                break

        return p, history

    def summarize(self, sentences: List[str], top_k: int = 4) -> Dict[str, Any]:
        """
        Runs the full TextRank pipeline on a list of sentences and returns both
        the extracted summary and the intermediate mathematical matrices/traces.
        """
        n = len(sentences)
        if n == 0:
            return {
                "summary_sentences": [],
                "ranked_sentences": [],
                "math_details": {}
            }

        top_k = max(1, min(top_k, n))
        tokenized = [preprocess_sentence(s, remove_stopwords=True, apply_stemming=True) for s in sentences]

        tfidf_matrix, vocab = self.build_tfidf_matrix(tokenized)
        sim_matrix = self.compute_similarity_matrix(tfidf_matrix)
        transition_matrix = self.build_transition_matrix(sim_matrix)
        pagerank_scores, convergence_history = self.power_iteration(transition_matrix)

        # Rank sentences by PageRank score descending
        ranked_indices = np.argsort(-pagerank_scores).tolist()
        selected_indices_set = set(ranked_indices[:top_k])

        # Preserve original chronological document order for coherent summary reading
        chronological_selected = sorted(list(selected_indices_set))

        summary_sentences = [
            {
                "sentence_index": idx,
                "text": sentences[idx],
                "pagerank_score": round(float(pagerank_scores[idx]), 5),
                "rank": ranked_indices.index(idx) + 1
            }
            for idx in chronological_selected
        ]

        all_ranked = [
            {
                "sentence_index": idx,
                "text": sentences[idx],
                "pagerank_score": round(float(pagerank_scores[idx]), 5),
                "rank": rank_pos + 1,
                "selected": idx in selected_indices_set
            }
            for rank_pos, idx in enumerate(ranked_indices)
        ]

        # Prepare compact preview of similarity & transition matrices (up to first 8 sentences)
        preview_n = min(n, 8)
        sim_preview = np.round(sim_matrix[:preview_n, :preview_n], 3).tolist()
        trans_preview = np.round(transition_matrix[:preview_n, :preview_n], 3).tolist()

        return {
            "summary_sentences": summary_sentences,
            "ranked_sentences": all_ranked,
            "math_details": {
                "algorithm": "TextRank (TF-IDF Cosine Graph + PageRank Power Iteration)",
                "damping_factor": self.damping_factor,
                "num_sentences": n,
                "vocabulary_size": len(vocab),
                "iterations_to_converge": convergence_history[-1]["iteration"] if convergence_history else 0,
                "final_l1_residual": convergence_history[-1]["l1_residual"] if convergence_history else 0.0,
                "convergence_history": convergence_history,
                "matrix_preview_size": preview_n,
                "similarity_matrix_preview": sim_preview,
                "transition_matrix_preview": trans_preview,
                "pagerank_vector": [round(float(x), 5) for x in pagerank_scores]
            }
        }
