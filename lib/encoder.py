"""The full pipeline: text → torus coordinates → winding analysis.

This is the bridge. It composes:
    1. Tokenize text into words
    2. Classify each token by grammatical role (→ interrogative component)
    3. Map to torus coordinates (θ₁, θ₂)
    4. Analyze winding numbers and spectral profile

The same torus FFT that finds periodicities in numeric signals
finds structural patterns in text. The data type is irrelevant
to the analysis — only the torus coordinates matter.
"""

from lib.grammar import tokenize, classify_token
from lib.components import Component, AssertionPhase
from lib.torus import to_torus, to_clifford
from lib.winding import (
    winding_numbers,
    spectral_profile,
    compare_profiles,
    compare_grammars,
)


class TorusPath:
    """A text encoded as a path on the flat torus."""

    def __init__(
        self,
        tokens: list[str],
        components: list[Component],
        phases: list[AssertionPhase],
        points: list[tuple[float, float]],
    ):
        self.tokens = tokens
        self.components = components
        self.phases = phases
        self.points = points

    def winding(self) -> tuple[float, float]:
        return winding_numbers(self.points)

    def profile(self) -> dict:
        return spectral_profile(self.points)

    def clifford_embedding(self) -> list[tuple[float, float, float, float]]:
        return [to_clifford(t1, t2) for t1, t2 in self.points]

    def component_histogram(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for c in self.components:
            counts[c.name] = counts.get(c.name, 0) + 1
        return counts

    def phase_histogram(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for p in self.phases:
            counts[p.name] = counts.get(p.name, 0) + 1
        return counts

    def transition_matrix(self) -> dict[tuple[str, str], int]:
        """Count transitions between consecutive (component, phase) states.

        Each state is the component name. Returns {(from, to): count}.
        The self-transition (A -> A) is the reflexive: the same instrument
        meeting itself. The cross-transition (A -> B) is the projected other.
        """
        counts: dict[tuple[str, str], int] = {}
        for i in range(len(self.components) - 1):
            key = (self.components[i].name, self.components[i + 1].name)
            counts[key] = counts.get(key, 0) + 1
        return counts

    def transition_entropy(self) -> float:
        """Shannon entropy of the transition distribution.

        High entropy = diverse transitions (the self meets many others).
        Low entropy = repetitive patterns (the self keeps meeting the same).
        """
        import math
        trans = self.transition_matrix()
        total = sum(trans.values())
        if total == 0:
            return 0.0
        ent = 0.0
        for count in trans.values():
            p = count / total
            if p > 0:
                ent -= p * math.log(p)
        return ent

    def self_transition_rate(self) -> float:
        """Fraction of transitions where the same component repeats.

        The reflexive rate: how often the instrument meets itself
        rather than projecting to an other.
        """
        trans = self.transition_matrix()
        total = sum(trans.values())
        if total == 0:
            return 0.0
        self_count = sum(v for (a, b), v in trans.items() if a == b)
        return self_count / total

    def windowed_entropy(self, window: int = 50) -> list[float]:
        """Transition entropy computed over a sliding window.

        Returns one entropy value per window position. Where entropy
        stabilizes, the coordinate functions have harmonized — the
        text's grammatical rhythm has settled. Where it keeps moving,
        the winding is still irrational.
        """
        import math
        n = len(self.components)
        if n < 2 or window < 2:
            return []
        results = []
        for start in range(n - window):
            counts: dict[tuple[str, str], int] = {}
            for i in range(start, start + window - 1):
                key = (self.components[i].name, self.components[i + 1].name)
                counts[key] = counts.get(key, 0) + 1
            total = sum(counts.values())
            ent = 0.0
            for count in counts.values():
                p = count / total
                if p > 0:
                    ent -= p * math.log(p)
            results.append(ent)
        return results

    def harmonization_curve(self, window: int = 50) -> list[float]:
        """Distance from local rhythm to global rhythm at each position.

        Measures |windowed_entropy - global_entropy| normalized by the
        global entropy. Values near 0 mean the local window matches the
        text's overall rhythm; values near 1 mean it diverges.
        """
        global_ent = self.transition_entropy()
        if global_ent == 0:
            return []
        return [abs(w - global_ent) / global_ent
                for w in self.windowed_entropy(window)]

    def rhythm_stability(self, window: int = 50) -> float:
        """How stable the grammatical rhythm is through the text.

        1.0 = perfectly stable (windowed entropy never varies).
        0.0 = maximally unstable.
        Computed as 1 - coefficient of variation of windowed entropy.
        """
        import statistics
        ents = self.windowed_entropy(window)
        if len(ents) < 2:
            return 1.0
        mean = statistics.mean(ents)
        if mean == 0:
            return 1.0
        cv = statistics.stdev(ents) / mean
        return max(0.0, 1.0 - cv)

    def harmonization_rate(self, window: int = 50) -> float:
        """How quickly the rhythm harmonizes from the text's opening.

        Measures the position (as fraction of text length) where the
        harmonization curve first drops below 0.1 (within 10% of global).
        Returns 1.0 if it never harmonizes; returns 0.0 if it starts harmonized.
        """
        curve = self.harmonization_curve(window)
        if not curve:
            return 0.0
        for i, val in enumerate(curve):
            if val < 0.1:
                return i / len(curve)
        return 1.0

    def __len__(self) -> int:
        return len(self.tokens)


def encode(text: str) -> TorusPath:
    """Encode text as a path on the torus.

    This is the bridge composition:
        text → tokens → (component, phase) → (θ₁, θ₂)
    """
    raw_tokens = tokenize(text)
    tokens = []
    components = []
    phases = []
    points = []

    for tok in raw_tokens:
        comp, phase = classify_token(tok)
        theta1, theta2 = to_torus(comp, phase)
        tokens.append(tok)
        components.append(comp)
        phases.append(phase)
        points.append((theta1, theta2))

    return TorusPath(tokens, components, phases, points)


def compare(text_a: str, text_b: str) -> dict:
    """Compare two texts by their grammatical structure on the torus.

    Returns both the grammatical distance (component fractions, phase
    fractions, winding rates — the features that separate genres) and
    the spectral profile comparison. The grammatical distance is the
    conjunction: two profiles placed side by side.
    """
    path_a = encode(text_a)
    path_b = encode(text_b)

    n_a, n_b = len(path_a), len(path_b)
    prof_a = path_a.profile()
    prof_b = path_b.profile()

    hist_a = path_a.component_histogram()
    hist_b = path_b.component_histogram()
    total_a = sum(hist_a.values())
    total_b = sum(hist_b.values())
    comp_fracs_a = {k: v / total_a for k, v in hist_a.items()} if total_a else {}
    comp_fracs_b = {k: v / total_b for k, v in hist_b.items()} if total_b else {}

    phase_a = path_a.phase_histogram()
    phase_b = path_b.phase_histogram()
    phase_fracs_a = {k: v / total_a for k, v in phase_a.items()} if total_a else {}
    phase_fracs_b = {k: v / total_b for k, v in phase_b.items()} if total_b else {}

    w_a = path_a.winding()
    w_b = path_b.winding()
    wr_a = (w_a[0] / n_a, w_a[1] / n_a) if n_a else (0.0, 0.0)
    wr_b = (w_b[0] / n_b, w_b[1] / n_b) if n_b else (0.0, 0.0)

    result = compare_grammars(
        comp_fracs_a, comp_fracs_b,
        phase_fracs_a, phase_fracs_b,
        wr_a, wr_b,
    )
    result.update(compare_profiles(prof_a, prof_b))
    result["profile_a"] = prof_a
    result["profile_b"] = prof_b

    return result
