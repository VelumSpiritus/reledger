"""Torus embedding — grammatical components to T² coordinates.

The state space is a flat torus: S¹ × S¹ = T².

Two axes:
    θ₁ (assertion axis): position in the query cycle
        opening → establishing → contrasting → closing
    θ₂ (instrument axis): which interrogative component
        whether / what / why / how / who / where-when / which / how-much

The torus is intrinsically flat (K ≡ 0). Curvature belongs to the
embedding, not the object.

Clifford embedding into R⁴:
    (θ₁, θ₂) → (cos θ₁, sin θ₁, cos θ₂, sin θ₂) / √2
"""

import math
from lib.components import (
    Component,
    AssertionPhase,
    INSTRUMENT_ANGLES,
    ASSERTION_ANGLES,
)


def to_torus(component: Component, phase: AssertionPhase) -> tuple[float, float]:
    """Map a classified token to torus coordinates (θ₁, θ₂).

    θ₁ = assertion phase angle (position in query cycle)
    θ₂ = instrument angle (which interrogative component)

    Both in [0, 2π).
    """
    theta1 = ASSERTION_ANGLES[phase]
    theta2 = INSTRUMENT_ANGLES[component]
    return (theta1, theta2)


def to_clifford(theta1: float, theta2: float) -> tuple[float, float, float, float]:
    """Embed torus point into R⁴ via the Clifford torus.

    The isometric embedding: (θ₁, θ₂) → (cos θ₁, sin θ₁, cos θ₂, sin θ₂) / √2

    This is the flat embedding — no curvature added by the room.
    """
    s = 1.0 / math.sqrt(2)
    return (
        s * math.cos(theta1),
        s * math.sin(theta1),
        s * math.cos(theta2),
        s * math.sin(theta2),
    )


def torus_distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    """Geodesic distance on the flat torus.

    The shortest path wrapping around either axis.
    """
    d1 = abs(p1[0] - p2[0])
    d2 = abs(p1[1] - p2[1])
    d1 = min(d1, 2 * math.pi - d1)
    d2 = min(d2, 2 * math.pi - d2)
    return math.sqrt(d1 * d1 + d2 * d2)


def angular_velocity(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Compute angular velocity between consecutive torus points.

    Returns (dθ₁, dθ₂) for each consecutive pair, wrapping correctly.
    """
    velocities = []
    for i in range(1, len(points)):
        d1 = points[i][0] - points[i - 1][0]
        d2 = points[i][1] - points[i - 1][1]
        # Wrap to [-π, π)
        while d1 > math.pi:
            d1 -= 2 * math.pi
        while d1 <= -math.pi:
            d1 += 2 * math.pi
        while d2 > math.pi:
            d2 -= 2 * math.pi
        while d2 <= -math.pi:
            d2 += 2 * math.pi
        velocities.append((d1, d2))
    return velocities
