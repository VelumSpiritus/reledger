"""Worked example: encoding text as a torus path and comparing structure.

Demonstrates the bridge: linguistic structure → grammatical components →
torus coordinates → winding analysis. The same FFT that finds
periodicities in numeric signals finds structural patterns in text.

Run from the repo root:
    python examples/encode_text.py
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.encoder import encode, compare
from lib.components import Component

# --- 1. Encode a single text ---

text = (
    "The researcher coded the transcripts carefully, but after three weeks "
    "the definitions started to blur. Because the codebook was ambiguous on "
    "edge cases, she found herself merging codes that should have stayed "
    "separate. If she had re-coded the calibration set weekly, the drift "
    "would have been caught earlier."
)

path = encode(text)

print("Text: %s..." % text[:60])
print("Tokens: %d" % len(path))
print()

# Component histogram — which grammatical components dominate
print("Component histogram (interrogative instruments):")
hist = path.component_histogram()
for name in sorted(hist, key=hist.get, reverse=True):
    bar = "#" * hist[name]
    print("  %-12s %3d  %s" % (name, hist[name], bar))
print()

# Phase histogram — where in the query cycle
print("Phase histogram (assertion axis):")
phases = path.phase_histogram()
for name in sorted(phases, key=phases.get, reverse=True):
    bar = "#" * phases[name]
    print("  %-14s %3d  %s" % (name, phases[name], bar))
print()

# Winding numbers
w1, w2 = path.winding()
print("Winding numbers:")
print("  Assertion axis (θ₁): %.3f" % w1)
print("  Instrument axis (θ₂): %.3f" % w2)
print()

# Spectral profile
prof = path.profile()
print("Top spectral frequencies:")
print("  Axis 1 (assertion):", [(f, "%.4f" % m) for f, m in prof["axis1_freqs"][:3]])
print("  Axis 2 (instrument):", [(f, "%.4f" % m) for f, m in prof["axis2_freqs"][:3]])
print()

# --- 2. Compare two structurally similar texts ---

text_a = (
    "The experiment failed because the temperature exceeded the threshold. "
    "Although the controls were properly calibrated, the sample degraded "
    "before measurement could begin."
)

text_b = (
    "The deployment crashed because the memory exceeded the limit. "
    "Although the tests were properly passing, the service degraded "
    "before users could connect."
)

text_c = (
    "42 samples, 18 controls, 7 outliers. The mean was 3.14 and the "
    "standard deviation 0.28. Most values fell within two sigma."
)

print("=" * 60)
print("Structural comparison")
print("=" * 60)
print()
print("Text A: %s..." % text_a[:50])
print("Text B: %s..." % text_b[:50])
print("Text C: %s..." % text_c[:50])
print()

ab = compare(text_a, text_b)
ac = compare(text_a, text_c)

print("A vs B (same structure, different domain):")
print("  Winding distance: %.4f" % ab["winding_distance"])
print("  Axis 1 spectral similarity: %.4f" % ab["axis1_spectral_similarity"])
print("  Axis 2 spectral similarity: %.4f" % ab["axis2_spectral_similarity"])
print()

print("A vs C (different structure entirely):")
print("  Winding distance: %.4f" % ac["winding_distance"])
print("  Axis 1 spectral similarity: %.4f" % ac["axis1_spectral_similarity"])
print("  Axis 2 spectral similarity: %.4f" % ac["axis2_spectral_similarity"])
print()

# --- 3. The bridge in action ---

print("=" * 60)
print("The bridge: grammatical components on the torus")
print("=" * 60)
print()
print("First 10 tokens of the sample text, classified:")
print()
for i, (tok, comp, phase) in enumerate(
    zip(path.tokens[:10], path.components[:10], path.phases[:10])
):
    theta1, theta2 = path.points[i]
    print(
        "  %-15s  → %-12s %-14s  (θ₁=%.2f, θ₂=%.2f)"
        % (tok, comp.name, phase.name, theta1, theta2)
    )
