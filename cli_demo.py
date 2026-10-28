"""
Standalone Command-Line Numerical Verification & Demo Runner.
Runs all 4 classical NLP algorithms step-by-step on a study chapter and prints
the intermediate mathematical matrices, probabilities, and final outputs.
"""

import json
import os
from nlp_engine.pipeline import StudyAssistantPipeline


def run_cli_demo():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    chapters_path = os.path.join(base_dir, "data", "sample_chapters.json")
    with open(chapters_path, "r", encoding="utf-8") as f:
        chapters = json.load(f)

    chapter = chapters[0]
    pipeline = StudyAssistantPipeline()

    print("=" * 80)
    print(f"CLASSICAL NLP STUDY ASSISTANT — NUMERICAL VERIFICATION")
    print(f"Document: {chapter['title']}")
    print("=" * 80)

    analysis = pipeline.analyze_document(chapter["text"], title=chapter["title"], summary_k=3)

    # 1. TextRank Output
    tr_math = analysis["summary"]["math_details"]
    print("\n[ALGORITHM 1: TEXTRANK (PageRank Power Iteration Summarization)]")
    print(f"  - Sentences (N): {tr_math['num_sentences']} | Vocabulary (|V|): {tr_math['vocabulary_size']}")
    print(f"  - Converged in {tr_math['iterations_to_converge']} iterations (L1 residual = {tr_math['final_l1_residual']})")
    print(f"  - Stationary PageRank Vector: {tr_math['pagerank_vector']}")
    print("  - Top Extracted Summary Sentences:")
    for s in analysis["summary"]["summary_sentences"]:
        print(f"      [S{s['sentence_index']+1}] (Score={s['pagerank_score']}): {s['text']}")

    # 2. HMM + Viterbi Output
    ie_data = analysis["information_extraction"]
    print("\n[ALGORITHM 2: HIDDEN MARKOV MODEL (HMM) + VITERBI SEQUENCE DECODING]")
    for etype, items in ie_data["entities_by_type"].items():
        names = [x["entity"] for x in items]
        print(f"  - Extracted {etype}: {names}")

    # 3 & 4. Okapi BM25 + Naive Bayes & Lesk Q&A
    print("\n[ALGORITHMS 3 & 4: OKAPI BM25 RETRIEVAL + NAIVE BAYES & LESK Q&A]")
    for q in chapter["preset_questions"]:
        res = pipeline.ask_question(q)
        qa = res["qa_lesk"]
        bm25_top = res["retrieval_bm25"]["top_passages"][0]
        nb_post = qa["math_details"]["naive_bayes"]["posterior_probabilities"]
        print("-" * 72)
        print(f"  Question            : {q}")
        print(f"  Naive Bayes EAT     : {qa['expected_answer_type']} (Posteriors: {nb_post})")
        print(f"  BM25 Top Passage    : [S{bm25_top['sentence_index']+1}] (BM25 Score = {bm25_top['bm25_score']})")
        print(f"  Pinpointed Answer   : {qa['exact_answer']}")
        if qa["lesk_disambiguations"]:
            lesk0 = qa["lesk_disambiguations"][0]
            print(f"  Lesk WSD ('{lesk0['word']}') : {lesk0['selected_sense']} (Overlap={lesk0['overlap_score']}, Stems={lesk0['overlapping_stems']})")


if __name__ == "__main__":
    run_cli_demo()
