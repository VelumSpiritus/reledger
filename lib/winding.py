"""Winding number analysis on torus paths via FFT.

A path on T² has two winding numbers: how many times it wraps around
each axis. The FFT decomposes the angular velocity into frequency
components — the dominant frequency on each axis is the winding rate.

This is the same analysis the torus-projector applies to numeric
time-series. The bridge makes the data type irrelevant: once
linguistic structure is mapped to torus coordinates, the same
FFT finds structural patterns in text that it finds periodicities
in signals.
"""

import math
import cmath


def _fft(x: list[complex]) -> list[complex]:
    """Radix-2 FFT. Input length must be a power of 2."""
    n = len(x)
    if n <= 1:
        return x
    even = _fft(x[0::2])
    odd = _fft(x[1::2])
    t = [cmath.exp(-2j * cmath.pi * k / n) * odd[k] for k in range(n // 2)]
    return [even[k] + t[k] for k in range(n // 2)] + \
           [even[k] - t[k] for k in range(n // 2)]


def _pad_to_power_of_2(seq: list[float]) -> list[float]:
    """Zero-pad to the next power of 2."""
    n = len(seq)
    if n == 0:
        return [0.0]
    p = 1
    while p < n:
        p *= 2
    return seq + [0.0] * (p - n)


def winding_numbers(points: list[tuple[float, float]]) -> tuple[float, float]:
    """Compute winding numbers of a path on T².

    The winding number on each axis is the total angular displacement
    divided by 2π. A path that wraps once around axis 1 has w₁ = 1.

    Returns (w₁, w₂) — the winding numbers on the assertion and
    instrument axes respectively.
    """
    if len(points) < 2:
        return (0.0, 0.0)

    total1 = 0.0
    total2 = 0.0
    for i in range(1, len(points)):
        d1 = points[i][0] - points[i - 1][0]
        d2 = points[i][1] - points[i - 1][1]
        while d1 > math.pi:
            d1 -= 2 * math.pi
        while d1 <= -math.pi:
            d1 += 2 * math.pi
        while d2 > math.pi:
            d2 -= 2 * math.pi
        while d2 <= -math.pi:
            d2 += 2 * math.pi
        total1 += d1
        total2 += d2

    return (total1 / (2 * math.pi), total2 / (2 * math.pi))


def spectral_profile(points: list[tuple[float, float]]) -> dict:
    """FFT-based spectral decomposition of a torus path.

    Returns the dominant frequencies and their magnitudes on each axis,
    plus the winding numbers. The spectral profile characterizes the
    structural rhythm of the text — how its grammatical components cycle.
    """
    if len(points) < 2:
        return {
            "winding": (0.0, 0.0),
            "axis1_freqs": [],
            "axis2_freqs": [],
        }

    # Extract angular velocities
    dtheta1 = []
    dtheta2 = []
    for i in range(1, len(points)):
        d1 = points[i][0] - points[i - 1][0]
        d2 = points[i][1] - points[i - 1][1]
        while d1 > math.pi:
            d1 -= 2 * math.pi
        while d1 <= -math.pi:
            d1 += 2 * math.pi
        while d2 > math.pi:
            d2 -= 2 * math.pi
        while d2 <= -math.pi:
            d2 += 2 * math.pi
        dtheta1.append(d1)
        dtheta2.append(d2)

    # FFT on each axis
    padded1 = _pad_to_power_of_2(dtheta1)
    padded2 = _pad_to_power_of_2(dtheta2)
    fft1 = _fft([complex(x) for x in padded1])
    fft2 = _fft([complex(x) for x in padded2])

    n = len(padded1)

    # Extract top frequencies (skip DC, take first half)
    def top_freqs(fft_result, count=5):
        mags = [(abs(fft_result[k]) / n, k) for k in range(1, n // 2)]
        mags.sort(reverse=True)
        return [(freq, mag) for mag, freq in mags[:count]]

    w = winding_numbers(points)

    return {
        "winding": w,
        "axis1_freqs": top_freqs(fft1),
        "axis2_freqs": top_freqs(fft2),
        "n_tokens": len(points),
    }


def compare_profiles(
    profile_a: dict, profile_b: dict
) -> dict:
    """Compare two spectral profiles.

    Returns a similarity measure: how structurally alike are the two
    texts in their grammatical rhythm on the torus.
    """
    w_a = profile_a["winding"]
    w_b = profile_b["winding"]

    # Winding distance
    dw1 = abs(w_a[0] - w_b[0])
    dw2 = abs(w_a[1] - w_b[1])
    winding_distance = math.sqrt(dw1 ** 2 + dw2 ** 2)

    # Spectral overlap: compare top frequency magnitudes
    def freq_vector(freqs, size=8):
        v = [0.0] * size
        for freq, mag in freqs:
            if freq < size:
                v[freq] = mag
        return v

    v1a = freq_vector(profile_a.get("axis1_freqs", []))
    v1b = freq_vector(profile_b.get("axis1_freqs", []))
    v2a = freq_vector(profile_a.get("axis2_freqs", []))
    v2b = freq_vector(profile_b.get("axis2_freqs", []))

    def cosine_sim(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        if na < 1e-12 or nb < 1e-12:
            return 0.0
        return dot / (na * nb)

    return {
        "winding_distance": winding_distance,
        "axis1_spectral_similarity": cosine_sim(v1a, v1b),
        "axis2_spectral_similarity": cosine_sim(v2a, v2b),
    }


def compare_grammars(
    comp_fracs_a: dict[str, float],
    comp_fracs_b: dict[str, float],
    phase_fracs_a: dict[str, float],
    phase_fracs_b: dict[str, float],
    winding_rate_a: tuple[float, float],
    winding_rate_b: tuple[float, float],
) -> dict:
    """Compare two texts by their grammatical profiles directly.

    Comparison is a conjunction: the two profiles placed side by side.
    Each feature that the ANOVA proved separates genres contributes
    to the distance. No spectral transform — the grammar IS the signal.
    """
    all_comps = sorted(set(comp_fracs_a) | set(comp_fracs_b))
    all_phases = sorted(set(phase_fracs_a) | set(phase_fracs_b))

    # Component fraction distance (Euclidean over the 8 instruments)
    comp_dist = math.sqrt(sum(
        (comp_fracs_a.get(c, 0.0) - comp_fracs_b.get(c, 0.0)) ** 2
        for c in all_comps
    ))

    # Phase fraction distance (Euclidean over the 4 assertion phases)
    phase_dist = math.sqrt(sum(
        (phase_fracs_a.get(p, 0.0) - phase_fracs_b.get(p, 0.0)) ** 2
        for p in all_phases
    ))

    # Winding rate distance
    wr_dist = math.sqrt(
        (winding_rate_a[0] - winding_rate_b[0]) ** 2 +
        (winding_rate_a[1] - winding_rate_b[1]) ** 2
    )

    # Combined: all three are Euclidean distances on their own scales
    # so the total is the norm over the full 14-dim profile vector
    combined = math.sqrt(comp_dist ** 2 + phase_dist ** 2 + wr_dist ** 2)

    return {
        "component_distance": comp_dist,
        "phase_distance": phase_dist,
        "winding_rate_distance": wr_dist,
        "grammatical_distance": combined,
    }
