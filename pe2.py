
from __future__ import annotations
import argparse
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
import json

import numpy as np
import matplotlib.pyplot as plt

# Global Constants
END_PUNCT = {".", "!", "?"}
DEFAULT_BOOK = Path(__file__).with_name("AliceinWonderland.txt")
MAX_NUM_OF_WORDS_IN_A_SENTENCE = 1500

# ---------------- Utility functions (copied, unmodified where instructed) ----------------
def load_book(path: Path) -> str:
    data = path.read_text(encoding="utf-8", errors="ignore")
    return strip_gutenberg_header_footer(data)

def strip_gutenberg_header_footer(text: str) -> str:
    start_pat = re.compile(r"\*\*\*\s*START OF (THIS|THE) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.I | re.S)
    end_pat   = re.compile(r"\*\*\*\s*END OF (THIS|THE) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.I | re.S)
    start = start_pat.search(text)
    end = end_pat.search(text)
    if start and end and start.end() < end.start():
        return text[start.end():end.start()]
    return text
    
def tokenize(text: str) -> List[str]:
    pattern = (
        r"(?:[A-Za-z]\.){2,}(?:[A-Za-z]\.)?"
        r"|[A-Za-z]+(?:['\u2019][A-Za-z]+)*"
        r"|(?<=[A-Za-z])[.!?]"
    )
    return [t.lower() for t in re.findall(pattern, text)]

def freq_of_words(tokens: List[str]) -> Dict[str, int]:
    freqs: Dict[str, int] = {}
    for t in tokens:
        if t in freqs:
            freqs[t] += 1
        else:
            freqs[t] = 1
    return freqs

def build_model(tokens: List[str]) -> Tuple[Dict[str, Dict[str, float]], List[str]]:
    transition_counter: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    start_words: List[str] = []
    prev_token = None

    for index, token in enumerate(tokens):
        if token not in END_PUNCT and (prev_token is None or prev_token in END_PUNCT):
            start_words.append(token)
        if token not in END_PUNCT and index < len(tokens) - 1:
            next_token = tokens[index + 1]
            transition_counter[token][next_token] += 1
        prev_token = token

    transition_probs: Dict[str, Dict[str, float]] = {}
    for word, next_counts in transition_counter.items():
        total = float(sum(next_counts.values()))
        transition_probs[word] = {nxt: c / total for nxt, c in next_counts.items()}

    return transition_probs, start_words

def weighted_choice(prob_dict):
    items = list(prob_dict.keys())
    weights = np.array([prob_dict[k] for k in items], dtype=float)
    return random.choices(items, weights=weights, k=1)[0]

def make_it_pretty(tokens: List[str]) -> str:
    pretty: List[str] = []
    for t in tokens:
        if t in END_PUNCT and pretty:
            pretty[-1] = pretty[-1] + t
        else:
            pretty.append(t)
    if pretty:
        pretty[0] = pretty[0].capitalize()
    return " ".join(pretty)

def generate_sentence_order_1(transition_probs, start_words_count, max_words=MAX_NUM_OF_WORDS_IN_A_SENTENCE) -> Tuple[str, int]:
    start_words_probs: Dict[str, float] = {}
    total = float(sum(start_words_count.values()))
    for word, count in start_words_count.items():
        start_words_probs[word] = count/total
    
    word = weighted_choice(start_words_probs)
    output = [word]
    words_emitted = 1

    for _ in range(max_words - 1):
        next_token = weighted_choice(transition_probs[word])
        output.append(next_token)
        if next_token not in END_PUNCT:
            words_emitted += 1
        if next_token in END_PUNCT:
            break
        word = next_token

    if output[-1] not in END_PUNCT:
        output.append(".")
    return make_it_pretty(output), words_emitted

# -------------------- REQUIRED IMPLEMENTATIONS --------------------

def generate_sentence_memoryless(tokens_count, max_words=MAX_NUM_OF_WORDS_IN_A_SENTENCE) -> Tuple[str, int]:
    """
    Order-0 (memoryless) generator using the global token frequency distribution.
    Returns (sentence, length_without_punct).
    """
    total = float(sum(tokens_count.values()))
    prob_dict: Dict[str, float] = {tok: cnt/total for tok, cnt in tokens_count.items()}
    out: List[str] = []
    words_emitted = 0
    for _ in range(max_words):
        token = weighted_choice(prob_dict)
        out.append(token)
        if token not in END_PUNCT:
            words_emitted += 1
        if token in END_PUNCT:
            break
    if not out or out[-1] not in END_PUNCT:
        out.append(".")
    return make_it_pretty(out), words_emitted

def question_one(tokens_count: Dict[str, int]) -> List[str]:
    """
    Q1: Stats + top30 + histogram + 10 memoryless sentences.
    Returns top_thirty_words.
    """
    punct_total = sum(cnt for tok, cnt in tokens_count.items() if tok in END_PUNCT)
    word_total  = sum(cnt for tok, cnt in tokens_count.items() if tok not in END_PUNCT)
    print(f"[Q1] Total words (excluding punctuation): {word_total}")
    print(f"[Q1] Total sentences (from punctuation): {punct_total}")
    avg_words_per_sentence = word_total / punct_total if punct_total else 0.0
    print(f"[Q1] Average words per sentence: {avg_words_per_sentence:.4f}")
    unique_once = [tok for tok, cnt in tokens_count.items() if tok not in END_PUNCT and cnt == 1]
    print(f"[Q1] Ten words that appear exactly once: {unique_once[:10]}")
    top_items = sorted(((tok, cnt) for tok, cnt in tokens_count.items() if tok not in END_PUNCT),
                       key=lambda x: x[1], reverse=True)[:30]
    top_thirty_words = [tok for tok, _ in top_items]
    print(f"[Q1] Top 30 words: {top_thirty_words}")
    freqs = [tokens_count[w] for w in top_thirty_words]
    plt.figure()
    plt.bar(range(len(top_thirty_words)), freqs)
    plt.xticks(range(len(top_thirty_words)), top_thirty_words, rotation=90)
    plt.tight_layout()
    plt.savefig(str(Path('figure1.pdf')), format='pdf')
    plt.close()
    print("[Q1-b] 10 sentences from memoryless model:")
    for i in range(10):
        s, L = generate_sentence_memoryless(tokens_count)
        print(f"    {i+1:02d}: {s} (len={L})")
    return top_thirty_words

def question_two(transition_probs: Dict[str, Dict[str, float]], start_words_count: Dict[str, int]):
    """
    Q2: Count last-word candidates and only-last words using transition_probs.
    """
    last_candidates = 0
    only_last = 0
    for word, nexts in transition_probs.items():
        next_keys = set(nexts.keys())
        if next_keys & END_PUNCT:
            last_candidates += 1
        if next_keys and next_keys <= END_PUNCT:
            only_last += 1
    print(f"[Q2] Words that can be last word: {last_candidates}")
    print(f"[Q2] Words that are ONLY last word: {only_last}")

def question_three(transition_probs: Dict[str, Dict[str, float]], 
                   start_words_count: Dict[str, int], top_thirty_words: List[str]):
    """
    Q3: (a) print 5 sentences; (b) generate corpora and histograms; (c) report discussion in PDF.
    """
    print("[Q3-a] Five sentences from order-1 Markov model:")
    for i in range(5):
        s, L = generate_sentence_order_1(transition_probs, start_words_count)
        print(f"    {i+1:02d}: {s} (len={L})")
    sizes = [5, 25, 100, 10000]
    fig_names = ['figure2.pdf','figure3.pdf','figure4.pdf','figure5.pdf']
    for i, fig in zip(sizes, fig_names):
        sentences = []
        total_words = 0
        for _ in range(i):
            s, L = generate_sentence_order_1(transition_probs, start_words_count)
            sentences.append(s)
            total_words += L
        avg_words = total_words / i if i else 0.0
        print(f"[Q3-b] i={i}: average words per sentence = {avg_words:.4f}")
        book_text = " ".join(sentences).lower()
        tokens = tokenize(book_text)
        gen_counts = freq_of_words(tokens)
        freqs = [gen_counts.get(w, 0) for w in top_thirty_words]
        plt.figure()
        plt.bar(range(len(top_thirty_words)), freqs)
        plt.xticks(range(len(top_thirty_words)), top_thirty_words, rotation=90)
        plt.tight_layout()
        plt.savefig(fig, format='pdf')
        plt.close()

def main():
    seed = random.randint(1, 10000)
    random.seed(seed)
    np.random.seed(seed)
    print(f"Seed is {seed}")
    raw = load_book(DEFAULT_BOOK)
    tokens_all = tokenize(raw)
    tokens_count = freq_of_words(tokens_all)
    transition_probs, start_words = build_model(tokens_all)
    start_words_count = freq_of_words(start_words)
    top_thirty_words = question_one(tokens_count)
    question_two(transition_probs, start_words_count)
    question_three(transition_probs, start_words_count, top_thirty_words)

if __name__ == "__main__":
    main()
