"""
Text Preprocessing Utilities for Classical NLP Algorithms.
Includes sentence segmentation, word tokenization, stopword filtering,
and a classical Porter-style suffix-stripping stemmer.
"""

import re
from typing import List, Tuple, Dict

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "because", "as", "until", "while",
    "of", "at", "by", "for", "with", "about", "against", "between", "into", "through",
    "during", "before", "after", "above", "below", "to", "from", "up", "down", "in",
    "out", "on", "off", "over", "under", "again", "further", "then", "once", "here",
    "there", "when", "where", "why", "how", "all", "any", "both", "each", "few",
    "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own",
    "same", "so", "than", "too", "very", "can", "will", "just", "should", "now",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having",
    "do", "does", "did", "doing", "would", "could", "ought", "i", "you", "he", "she",
    "it", "we", "they", "them", "their", "his", "her", "its", "our", "your", "this",
    "that", "these", "those", "which", "who", "whom", "whose", "what"
}

ABBREVIATIONS = {
    "dr.", "mr.", "mrs.", "ms.", "prof.", "sr.", "jr.", "st.", "vs.", "etc.",
    "e.g.", "i.e.", "fig.", "al.", "inc.", "ltd.", "co.", "jan.", "feb.", "mar.",
    "apr.", "aug.", "sept.", "oct.", "nov.", "dec.", "u.s.", "u.k."
}


def split_sentences(text: str) -> List[str]:
    """
    Splits raw document text into clean individual sentences while preserving
    common abbreviations (e.g., Dr., Prof., U.S., e.g.) and decimal numbers.
    """
    if not text or not text.strip():
        return []

    # Normalize whitespace while keeping paragraph breaks as sentence boundaries
    cleaned = re.sub(r'\r\n', '\n', text.strip())
    # Split paragraphs first
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n+', cleaned) if p.strip()]

    sentences: List[str] = []
    for para in paragraphs:
        # Collapse single newlines within a paragraph
        para_line = re.sub(r'\s+', ' ', para).strip()
        # Tokenize by candidate sentence boundaries
        tokens = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9"\'\(\[])', para_line)
        buffer = ""
        for tok in tokens:
            if buffer:
                candidate = buffer + " " + tok
            else:
                candidate = tok

            # Check if candidate ends with an abbreviation or single initial (like "A.")
            last_word = candidate.split()[-1].lower() if candidate.split() else ""
            is_initial = bool(re.match(r'^[a-z]\.$', last_word))
            if last_word in ABBREVIATIONS or is_initial:
                buffer = candidate
            else:
                if len(candidate.strip()) > 3:
                    sentences.append(candidate.strip())
                buffer = ""
        if buffer and len(buffer.strip()) > 3:
            sentences.append(buffer.strip())

    return sentences


def tokenize_words(text: str, lowercase: bool = True) -> List[str]:
    """
    Extracts word/number tokens from text.
    """
    if lowercase:
        text = text.lower()
    return re.findall(r"[A-Za-z0-9]+(?:['\-][A-Za-z0-9]+)*", text)


def tokenize_with_punctuation(sentence: str) -> List[str]:
    """
    Tokenizes a sentence preserving case for HMM sequence tagging.
    """
    return re.findall(r"[A-Za-z0-9]+(?:['\-\.][A-Za-z0-9]+)*|[.,!?;:()\"']", sentence)


def stem_word(word: str) -> str:
    """
    Classical rule-based suffix-stripping stemmer (Porter-inspired)
    to normalize morphological variants without external black-box libraries.
    """
    w = word.lower().strip()
    if len(w) <= 3:
        return w

    # Step 1: Plurals and past/progressive participles
    if w.endswith("ies") and len(w) > 4:
        w = w[:-3] + "y"
    elif w.endswith("es") and len(w) > 4 and w[-3] in "sxz" or w.endswith(("ches", "shes")):
        w = w[:-2]
    elif w.endswith("s") and not w.endswith(("ss", "us", "is")) and len(w) > 3:
        w = w[:-1]

    if w.endswith("ing") and len(w) > 5:
        base = w[:-3]
        if len(base) >= 3 and base[-1] == base[-2] and base[-1] not in "aeiou":
            w = base[:-1]
        else:
            w = base
    elif w.endswith("ed") and len(w) > 4:
        base = w[:-2]
        if len(base) >= 3 and base[-1] == base[-2] and base[-1] not in "aeiou":
            w = base[:-1]
        else:
            w = base

    # Step 2: Derivational suffixes
    suffix_replacements = [
        ("ational", "ate"),
        ("tional", "tion"),
        ("ization", "ize"),
        ("iveness", "ive"),
        ("fulness", "ful"),
        ("ousness", "ous"),
        ("biliti", "ble"),
        ("icate", "ic"),
        ("ative", ""),
        ("alize", "al"),
        ("iciti", "ic"),
        ("ical", "ic"),
        ("ness", ""),
        ("ment", ""),
        ("able", ""),
        ("ible", ""),
        ("ally", "al"),
        ("ly", ""),
    ]
    for suffix, repl in suffix_replacements:
        if w.endswith(suffix) and len(w) - len(suffix) >= 3:
            w = w[:-len(suffix)] + repl
            break

    return w


def preprocess_sentence(sentence: str, remove_stopwords: bool = True, apply_stemming: bool = True) -> List[str]:
    """
    Full preprocessing pipeline for a sentence:
    Tokenization -> Lowercasing -> Stopword Removal -> Stemming.
    """
    tokens = tokenize_words(sentence, lowercase=True)
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    if apply_stemming:
        tokens = [stem_word(t) for t in tokens]
    return tokens
