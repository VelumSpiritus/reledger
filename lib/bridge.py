"""Bridges: raw data → (N, 8) feature arrays.

The bridge is the ONLY data-type-specific piece of the pipeline.
Everything after the bridge — soft encoding, revolution cycle,
silhouette, genre discovery — is identical regardless of data type.

Two bridges ship with the package:
    bridge_signal  — any 1D numerical signal (vibration, audio, EEG, …)
    bridge_text    — natural-language text via grammatical classification

Both return an (N, 8) numpy array: N windows, 8 structural features.

Semantic return:
    collect_open_class — per-window open-class tokens, aligned with
        bridge_text output. Feed to semantics.genre_semantics() after
        genre discovery to get the return trip: structure → meaning.
"""

import numpy as np

from lib.engine import COMPONENT_NAMES


BRIDGE_MIN_WINDOW = 4

def bridge_signal(signal, window_size=64):
    """Universal signal bridge.

    Extracts 8 structural features from each window of a 1D numerical
    signal. Works for vibration, audio, EEG, or anything that can be
    windowed. The features are structural properties of the window —
    they don't depend on knowing what the signal is.

    When the signal is shorter than window_size, the window adapts down
    to BRIDGE_MIN_WINDOW (the floor set by the semantic operations:
    acceleration needs 3 samples, spectral thirds need 4).

    Parameters
    ----------
    signal : array-like
        1D numerical signal.
    window_size : int
        Number of samples per window.

    Returns
    -------
    np.ndarray, shape (N, 8)
        Feature array with N windows.
    """
    signal = np.asarray(signal, dtype=float)
    if len(signal) < window_size:
        window_size = max(BRIDGE_MIN_WINDOW, len(signal))
    if len(signal) < BRIDGE_MIN_WINDOW:
        return np.zeros((0, 8))
    sig = (signal - np.mean(signal)) / (np.std(signal) + 1e-10)
    out = []
    for start in range(0, len(signal) - window_size + 1, window_size):
        w = sig[start:start + window_size]
        amp = np.abs(w).mean()
        vel = np.abs(np.diff(w)).mean()
        dc = abs(w.mean())
        energy = np.mean(w ** 2)
        phase = abs(np.mean(w > 0) - 0.5) * 2
        accel = np.abs(np.diff(w, n=2)).mean() if len(w) > 2 else 0
        zc = np.sum(np.abs(np.diff(np.sign(w))) > 0) / window_size
        fft_mag = np.abs(np.fft.rfft(w))
        nb = len(fft_mag)
        t = nb // 3
        lo = fft_mag[1:t + 1].sum()
        mi = fft_mag[t + 1:2 * t + 1].sum()
        hi = fft_mag[2 * t + 1:].sum()
        sd = max(lo, mi, hi) / (lo + mi + hi + 1e-10)
        out.append([amp, vel, dc, energy, phase, accel, zc, sd])
    return np.array(out) if out else np.zeros((0, 8))


def bridge_text(text, window_size=50):
    """Text bridge via grammatical classification.

    Counts occurrences of each of the 8 interrogative components
    within sliding windows of tokens. Uses the grammar module
    for classification.

    Parameters
    ----------
    text : str
        Natural-language text.
    window_size : int
        Number of tokens per window.

    Returns
    -------
    np.ndarray, shape (N, 8)
        Feature array with N windows, each a normalized component
        distribution.
    """
    from lib.grammar import classify_token, tokenize

    tokens = tokenize(text)
    comp_map = {c: i for i, c in enumerate(COMPONENT_NAMES)}
    rows = []
    for start in range(0, len(tokens) - window_size, window_size):
        window = tokens[start:start + window_size]
        counts = np.zeros(8)
        for tok in window:
            comp, _phase = classify_token(tok)
            idx = comp_map.get(comp.name)
            if idx is not None:
                counts[idx] += 1
        if counts.sum() > 0:
            rows.append(counts / counts.sum())
        else:
            rows.append(np.ones(8) / 8)
    return np.array(rows) if rows else np.zeros((0, 8))


def collect_open_class(text, window_size=50):
    """Collect open-class tokens per window, aligned with bridge_text.

    Returns the content words (nouns, verbs, adjectives, adverbs) from
    each window. The window boundaries match bridge_text exactly, so
    index i here corresponds to row i of bridge_text output.

    Parameters
    ----------
    text : str
        Natural-language text.
    window_size : int
        Number of tokens per window (must match bridge_text).

    Returns
    -------
    list[list[str]]
        Open-class tokens per window.
    """
    from lib.grammar import is_open_class, tokenize

    tokens = tokenize(text)
    windows = []
    for start in range(0, len(tokens) - window_size, window_size):
        window = tokens[start:start + window_size]
        windows.append([tok for tok in window if is_open_class(tok)])
    return windows


def collect_all_tokens(text, window_size=50):
    """Collect all tokens per window, aligned with bridge_text.

    Returns every token from each window — closed-class and open-class
    both. Window boundaries match bridge_text and collect_open_class
    exactly, so index i here corresponds to row i of bridge_text output.

    Parameters
    ----------
    text : str
        Natural-language text.
    window_size : int
        Number of tokens per window (must match bridge_text).

    Returns
    -------
    list[list[str]]
        All tokens per window.
    """
    from lib.grammar import tokenize

    tokens = tokenize(text)
    windows = []
    for start in range(0, len(tokens) - window_size, window_size):
        windows.append(tokens[start:start + window_size])
    return windows
