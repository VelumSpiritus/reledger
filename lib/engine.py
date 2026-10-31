"""Data-agnostic torus engine.

Everything in this module works on (N, 8) feature arrays regardless
of where they came from — text, vibration, audio, EEG, genomics.
The bridge (in bridge.py) is the only data-type-specific piece;
this module is what runs after it.

Core pipeline:
    features (N, 8) → softmax_batch → revolution_cycle → find_natural_k

CC0 1.0 Universal — public domain.
"""

import math
from collections import Counter

import numpy as np

COMPONENT_NAMES = [
    'WHAT', 'HOW', 'WHO', 'HOWMUCH',
    'WHEREWHEN', 'WHY', 'WHETHER', 'WHICH',
]

COMPONENT_MEANINGS = {
    'WHAT': 'magnitude / amplitude',
    'HOW': 'rate of change / velocity',
    'WHO': 'baseline identity / DC offset',
    'HOWMUCH': 'energy / power',
    'WHEREWHEN': 'phase position / timing',
    'WHY': 'curvature / acceleration',
    'WHETHER': 'oscillation pattern / zero-crossings',
    'WHICH': 'frequency content / spectral shape',
}

COMPONENT_GRAMMAR = {
    'WHAT': 'content nouns, connective conjunctions (and/also), contrastive (but/however)',
    'HOW': 'auxiliaries (is/was/can/must), verbs by tense (-ed past, -ing progressive)',
    'WHO': 'pronouns: subject (I/she), possessive (my/his), object (me/him), reflexive (myself)',
    'HOWMUCH': 'quantifiers (many/few/all/some/every), numbers, negator (none)',
    'WHEREWHEN': 'location prepositions (in/on/at), temporal conjunctions (when/before/after)',
    'WHY': 'causal conjunctions (because/therefore/since/hence/so/thus)',
    'WHETHER': 'conditionals (if/unless), punctuation (. ! ? , ;), negation (not)',
    'WHICH': 'selective conjunctions (or/either/neither), interrogative (which)',
}


def softmax_batch(X, temperature=0.25):
    """Vectorized softmax over rows of a 2D array.

    Parameters
    ----------
    X : np.ndarray, shape (N, 8)
        Raw feature values.
    temperature : float
        Softmax temperature. Lower values produce sharper distributions.

    Returns
    -------
    np.ndarray, shape (N, 8)
        Row-wise softmax probabilities.
    """
    X = np.asarray(X)
    maxes = X.max(axis=1, keepdims=True)
    e = np.exp((X - maxes) / temperature)
    return e / e.sum(axis=1, keepdims=True)


def markov_entropy_rate(tokens, n_states=8):
    """Entropy rate of a first-order Markov chain over token indices.

    Parameters
    ----------
    tokens : list[int]
        Sequence of state indices (0–7).
    n_states : int
        Number of states.

    Returns
    -------
    float
        Bits per transition.
    """
    trans = np.zeros((n_states, n_states))
    for i in range(len(tokens) - 1):
        trans[tokens[i], tokens[i + 1]] += 1
    rs = trans.sum(axis=1)
    P = np.zeros_like(trans)
    for i in range(n_states):
        if rs[i] > 0:
            P[i] = trans[i] / rs[i]
    pi = rs / (rs.sum() + 1e-10)
    H = 0.0
    for i in range(n_states):
        if pi[i] > 0:
            for j in range(n_states):
                if P[i, j] > 0:
                    H -= pi[i] * P[i, j] * math.log2(P[i, j])
    return H


def js_divergence(p, q):
    """Jensen-Shannon divergence between two distributions.

    Parameters
    ----------
    p, q : np.ndarray
        Probability distributions (same length, summing to 1).

    Returns
    -------
    float
        JSD in nats.
    """
    p, q = np.asarray(p), np.asarray(q)
    m = (p + q) / 2
    return (0.5 * np.sum(p * np.log(p / (m + 1e-10) + 1e-10))
            + 0.5 * np.sum(q * np.log(q / (m + 1e-10) + 1e-10)))


def silhouette_score(features, labels):
    """Silhouette score for a labelled feature set.

    Parameters
    ----------
    features : np.ndarray, shape (N, D)
        Feature vectors.
    labels : list
        Cluster label per feature vector.

    Returns
    -------
    float
        Mean silhouette coefficient in [-1, 1].
    """
    unique = sorted(set(labels))
    if len(unique) < 2:
        return 0.0
    scores = []
    for i in range(len(features)):
        own = [j for j in range(len(features))
               if labels[j] == labels[i] and j != i]
        if not own:
            scores.append(0.0)
            continue
        a_i = np.mean([np.linalg.norm(features[i] - features[j])
                       for j in own])
        b_i = float('inf')
        for lab in unique:
            if lab == labels[i]:
                continue
            other = [j for j in range(len(features)) if labels[j] == lab]
            if other:
                d = np.mean([np.linalg.norm(features[i] - features[j])
                             for j in other])
                b_i = min(b_i, d)
        denom = max(a_i, b_i)
        scores.append((b_i - a_i) / denom if denom > 0 else 0)
    return float(np.mean(scores))


def _same_partition(a, b):
    """Check whether two label sequences define the same partition."""
    n = min(len(a), len(b))
    for i in range(n):
        for j in range(i + 1, n):
            if (a[i] == a[j]) != (b[i] == b[j]):
                return False
    return True


def revolution_cycle(features, labels, target_k, max_revs=20):
    """Run the revolution cycle: cluster → re-classify → repeat.

    Agglomerative clustering merges the two closest genre profiles
    (by Jensen-Shannon divergence), then every file is re-classified
    to its nearest centroid. Repeats until the partition converges.

    Parameters
    ----------
    features : np.ndarray, shape (N, D)
        File-level feature vectors.
    labels : list
        Initial label per file (e.g. source name).
    target_k : int
        Target number of genres.
    max_revs : int
        Maximum revolution iterations.

    Returns
    -------
    (labels, n_revs) : (list, int)
        Converged labels and number of revolutions taken.
    """
    current = list(labels)
    prev = None

    for rev in range(1, max_revs + 1):
        unique_labs = sorted(set(current))
        nl = len(unique_labs)
        if nl <= target_k:
            break

        lab_profiles = {}
        for lab in unique_labs:
            idxs = [i for i, l in enumerate(current) if l == lab]
            lab_profiles[lab] = np.mean(features[idxs], axis=0)

        lab_list = sorted(lab_profiles.keys())
        js_mat = {}
        for i, li in enumerate(lab_list):
            for j, lj in enumerate(lab_list):
                if i != j:
                    js_mat[(li, lj)] = js_divergence(
                        lab_profiles[li], lab_profiles[lj])

        clusters = {lab: [lab] for lab in lab_list}
        active = set(lab_list)
        while len(active) > target_k:
            best_d = float('inf')
            best_p = None
            active_list = sorted(active)
            for ii, a in enumerate(active_list):
                for b in active_list[ii + 1:]:
                    d_sum = sum(js_mat.get((bi, bj), 0)
                                for bi in clusters[a]
                                for bj in clusters[b])
                    d_avg = d_sum / (len(clusters[a]) * len(clusters[b]))
                    if d_avg < best_d:
                        best_d = d_avg
                        best_p = (a, b)
            a, b = best_p
            new_name = f"{a}+{b}"
            clusters[new_name] = clusters[a] + clusters[b]
            active.remove(a)
            active.remove(b)
            active.add(new_name)

        fam_centroids = {}
        for fam_name in active:
            idxs = [i for i, l in enumerate(current)
                    if l in clusters[fam_name]]
            if idxs:
                fam_centroids[fam_name] = np.mean(features[idxs], axis=0)

        new_labels = []
        for i in range(len(features)):
            best = min(fam_centroids,
                       key=lambda f: np.linalg.norm(
                           features[i] - fam_centroids[f]))
            new_labels.append(best)

        if prev is not None and _same_partition(new_labels, prev):
            return new_labels, rev
        prev = new_labels
        current = new_labels

    return current, rev


def find_natural_k(features, source_labels, max_k=None):
    """Find the natural genre count via peak silhouette score.

    Runs revolution_cycle at each candidate k and picks the k whose
    converged partition has the highest silhouette.

    Parameters
    ----------
    features : np.ndarray, shape (N, D)
        File-level feature vectors.
    source_labels : list
        Initial label per file (e.g. source name).
    max_k : int or None
        Maximum k to try. Defaults to min(n_sources, 10).

    Returns
    -------
    (best, all_results)
        best : (k, silhouette, labels, n_revs)
        all_results : list of (k, silhouette, labels, n_revs) for each k
    """
    n_sources = len(set(source_labels))
    if max_k is None:
        max_k = min(n_sources, 10)

    results = []
    for k in range(2, max_k + 1):
        labels_k, revs = revolution_cycle(features, source_labels, k)
        sil = silhouette_score(features, labels_k)
        results.append((k, sil, labels_k, revs))

    best = max(results, key=lambda r: r[1])
    return best, results
