"""The eight structural components — closed-class feature channels.

Every natural language carries a finite, closed class of grammatical
function words (Talmy, 2000). The engine decomposes these into eight
orthogonal feature channels, each capturing a distinct structural role.

The set is closed and universal: children acquire it before content words.
"""

from enum import Enum, auto
import math


class Component(Enum):
    C0 = 0   # binary cut
    C1 = 1   # referent naming
    C2 = 2   # causal direction
    C3 = 3   # process / manner
    C4 = 4   # participant identity
    C5 = 5   # spatiotemporal position
    C6 = 6   # reference resolution
    C7 = 7   # quantity / degree


# Fixed angular positions on the instrument axis (θ₂).
# Eight components evenly partition the circle.
INSTRUMENT_ANGLES = {
    Component.C0: 0.0,
    Component.C1: math.pi / 4,
    Component.C2: math.pi / 2,
    Component.C3: 3 * math.pi / 4,
    Component.C4: math.pi,
    Component.C5: 5 * math.pi / 4,
    Component.C6: 3 * math.pi / 2,
    Component.C7: 7 * math.pi / 4,
}


class AssertionPhase(Enum):
    """Position on the assertion axis (θ₁) — where in the query cycle."""
    OPENING = 0       # posing the question
    ESTABLISHING = 1  # covering, asserting
    CONTRASTING = 2   # negating, turning
    CLOSING = 3       # concluding, handing forward


ASSERTION_ANGLES = {
    AssertionPhase.OPENING:      0.0,
    AssertionPhase.ESTABLISHING: math.pi / 2,
    AssertionPhase.CONTRASTING:  math.pi,
    AssertionPhase.CLOSING:      3 * math.pi / 2,
}
