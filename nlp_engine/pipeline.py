"""
Unified Classical NLP Pipeline Orchestrator.
Chains all 4 distinct classical NLP algorithms on any user-provided or preset document:
  1. TextRank (PageRank Power Iteration) -> Extractive Executive Summary
  2. HMM + Viterbi Decoding -> Named Entity & Concept Sequence Tagging + Study Flashcards
  3. Okapi BM25 -> Probabilistic Passage Retrieval for User Questions
  4. Multinomial Naive Bayes + Simplified Lesk WSD -> Expected Answer Type Classification,
     Exact Span Pinpointing, and Word Sense Disambiguation
"""

from typing import Dict, Any, List
from .preprocessor import split_sentences
from .textrank import TextRankSummarizer
from .hmm_viterbi import HMMViterbiExtractor
from .bm25 import OkapiBM25Retriever
from .qa_naive_bayes_lesk import NaiveBayesLeskQA


class StudyAssistantPipeline:
    def __init__(self):
        self.textrank = TextRankSummarizer(damping_factor=0.85, max_iter=100, tol=1e-6)
        self.hmm_viterbi = HMMViterbiExtractor()
        self.bm25 = OkapiBM25Retriever(k1=1.5, b=0.75)
        self.qa_engine = NaiveBayesLeskQA()

        # State for the currently loaded document
        self.current_title: str = ""
        self.current_text: str = ""
        self.sentences: List[str] = []
        self.last_summary_result: Dict[str, Any] = {}
        self.last_ie_result: Dict[str, Any] = {}

    def analyze_document(
        self,
        text: str,
        title: str = "Custom Study Document",
        summary_k: int = 4,
        damping_factor: float = 0.85
    ) -> Dict[str, Any]:
        """
        Processes any raw document text through:
          - Sentence Segmentation
          - Algorithm 1: TextRank Summarizer
          - Algorithm 2: HMM + Viterbi Entity & Concept Extractor
          - Algorithm 3: Okapi BM25 Index Construction
        """
        self.current_title = title
        self.current_text = text
        self.sentences = split_sentences(text)

        self.textrank.damping_factor = damping_factor
        self.last_summary_result = self.textrank.summarize(self.sentences, top_k=summary_k)
        self.last_ie_result = self.hmm_viterbi.extract_entities_and_flashcards(self.sentences)
        self.bm25.index_sentences(self.sentences)

        # Generate suggested questions automatically from the extracted entities so
        # even when a user uploads a brand-new custom document, they get instant 1-click questions!
        auto_questions = self._generate_suggested_questions(self.last_ie_result["entities_by_type"])

        return {
            "title": self.current_title,
            "num_sentences": len(self.sentences),
            "sentences": [
                {
                    "index": idx,
                    "text": s,
                    "annotations": self.last_ie_result["sentence_annotations"][idx]
                    if idx < len(self.last_ie_result["sentence_annotations"]) else {}
                }
                for idx, s in enumerate(self.sentences)
            ],
            "summary": self.last_summary_result,
            "information_extraction": {
                "entities_by_type": self.last_ie_result["entities_by_type"],
                "flashcards": self.last_ie_result["flashcards"],
                "math_details": self.last_ie_result["math_details"]
            },
            "suggested_questions": auto_questions
        }

    def ask_question(
        self,
        question: str,
        text: str = None,
        k1: float = 1.5,
        b: float = 0.75
    ) -> Dict[str, Any]:
        """
        Answers a natural language question over the active document (or newly passed text)
        by chaining:
          - Algorithm 3: Okapi BM25 Passage Retrieval
          - Algorithm 2 Annotations: HMM + Viterbi Entity Spans inside the retrieved passages
          - Algorithm 4: Multinomial Naive Bayes EAT Classification + Span Pinpointing + Lesk WSD
        """
        if text and text.strip() and text.strip() != self.current_text.strip():
            self.analyze_document(text)

        self.bm25.k1 = k1
        self.bm25.b = b

        bm25_result = self.bm25.retrieve(question, top_k=3)
        qa_result = self.qa_engine.answer_question(
            question=question,
            bm25_result=bm25_result,
            sentence_annotations=self.last_ie_result.get("sentence_annotations", []),
            all_sentences=self.sentences
        )

        return {
            "question": question,
            "retrieval_bm25": bm25_result,
            "qa_lesk": qa_result
        }

    def _generate_suggested_questions(self, entities_by_type: Dict[str, List[Dict[str, Any]]]) -> List[str]:
        """
        Dynamically builds relevant study questions for ANY uploaded document
        based on the HMM-Viterbi extracted entities.
        """
        suggestions: List[str] = []

        persons = entities_by_type.get("PERSON", [])
        dates = entities_by_type.get("DATE", [])
        locations = entities_by_type.get("LOCATION", [])
        concepts = entities_by_type.get("CONCEPT", [])

        if persons:
            p = persons[0]["entity"]
            suggestions.append(f"Who was {p} and what did they contribute?")
        if dates and concepts:
            c = concepts[0]["entity"]
            suggestions.append(f"When was {c} developed or discovered?")
        elif dates:
            d = dates[0]["entity"]
            suggestions.append(f"What happened in {d}?")
        if locations:
            loc = locations[0]["entity"]
            suggestions.append(f"Where did events involving {loc} take place?")
        if concepts:
            for c_item in concepts[:2]:
                c_name = c_item["entity"]
                q = f"What is {c_name} and how does it work?"
                if q not in suggestions:
                    suggestions.append(q)

        return suggestions[:5]
