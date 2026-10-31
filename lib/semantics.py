"""Semantic return — structure finds genres, vocabulary tells you what they mean.

The torus engine strips semantics to find structural genres. This module
wires them back: after genre discovery, it aggregates each genre's
open-class vocabulary and looks up definitions in a dictionary.

Pipeline:
    text → bridge_text (structure) + collect_open_class (vocabulary)
    features → find_natural_k → genre labels
    labels + vocabulary + dictionary → genre_semantics → meaning per genre

The dictionary is user-supplied (any {word: definition} JSON). Webster's
1913 (public domain, 102k entries) is the default reference.

CC0 1.0 Universal — public domain.
"""

import json
from collections import Counter


def load_dictionary(path):
    """Load a word→definition dictionary from a JSON file.

    Parameters
    ----------
    path : str or Path
        Path to a JSON file mapping words to definitions.
        Expected format: {"word": "definition", ...}

    Returns
    -------
    dict[str, str]
        Lowercased word → definition mapping.
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return {k.lower(): v for k, v in raw.items()}


def genre_semantics(labels, open_class_windows, dictionary=None, top_n=10):
    """Return characteristic vocabulary and definitions for each genre.

    After the engine groups windows into genres by structure alone, this
    function aggregates the open-class tokens per genre and (optionally)
    looks up their definitions — the semantic return trip.

    Parameters
    ----------
    labels : list
        Genre label per window (from find_natural_k or revolution_cycle).
    open_class_windows : list[list[str]]
        Open-class tokens per window (from collect_open_class).
    dictionary : dict[str, str] or None
        Word → definition mapping. If None, definitions are omitted.
    top_n : int
        Number of most frequent tokens to return per genre.

    Returns
    -------
    dict
        Per genre: {"top": [(word, count), ...],
                    "definitions": {word: definition},
                    "total_tokens": int,
                    "unique_tokens": int}
    """
    genre_tokens = {}
    for label, tokens in zip(labels, open_class_windows):
        if label not in genre_tokens:
            genre_tokens[label] = []
        genre_tokens[label].extend(t.lower() for t in tokens)

    result = {}
    for label, tokens in genre_tokens.items():
        counts = Counter(tokens)
        top = counts.most_common(top_n)
        defs = {}
        if dictionary is not None:
            for word, _ in top:
                if word in dictionary:
                    defs[word] = dictionary[word]
        result[label] = {
            "top": top,
            "definitions": defs,
            "total_tokens": len(tokens),
            "unique_tokens": len(counts),
        }
    return result


def reledger(labels, token_windows, file_names=None):
    """Re-ledger content by discovered genre, preserving original order.

    Groups the original token windows by their genre label. Within each
    genre, windows appear in their original sequence — the content
    reorganized by structure, not summarized.

    Parameters
    ----------
    labels : list
        Genre label per window (from find_natural_k or revolution_cycle).
    token_windows : list[list[str]]
        Tokens per window (from collect_all_tokens or collect_open_class).
    file_names : list[str] or None
        Source filename per window, for tracing.

    Returns
    -------
    dict
        Per genre: {"windows": [token lists in original order],
                    "indices": [original window indices],
                    "sources": [source filename per window] or []}
    """
    genres = {}
    for i, label in enumerate(labels):
        if label not in genres:
            genres[label] = {"windows": [], "indices": [], "sources": []}
        if i < len(token_windows):
            genres[label]["windows"].append(token_windows[i])
        genres[label]["indices"].append(i)
        if file_names and i < len(file_names):
            genres[label]["sources"].append(file_names[i])
    return genres
