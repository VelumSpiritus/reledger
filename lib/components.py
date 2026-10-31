"""The eight interrogative components — closed-class grammatical instruments.

Every natural language issues the same eight question-forming instruments
(Talmy, 2000). Each maps to a structural role:

    what       — names the referent (cover)
    why        — establishes cause or purpose (direction)
    how        — carries the process or manner (motion)
    who        — identifies the participant (subject/object)
    where/when — marks position in space or time
    which      — resolves underdetermined reference into function
    how much   — measures quantity or degree (grain)
    whether    — the bare yes/no cut (primordial — every question presupposes it)

The set is closed and universal: children acquire it before content words.
"""

from enum import Enum, auto
import math


class Component(Enum):
    WHETHER = 0   # yes/no — primordial
    WHAT = 1      # names the referent
    WHY = 2       # cause or purpose
    HOW = 3       # process or manner
    WHO = 4       # participant identity
    WHEREWHEN = 5 # position in space/time
    WHICH = 6     # resolves underdetermined reference into function
    HOWMUCH = 7   # quantity or degree


# Fixed angular positions on the instrument axis (θ₂).
# Eight components evenly partition the circle.
INSTRUMENT_ANGLES = {
    Component.WHETHER:   0.0,
    Component.WHAT:      math.pi / 4,
    Component.WHY:       math.pi / 2,
    Component.HOW:       3 * math.pi / 4,
    Component.WHO:       math.pi,
    Component.WHEREWHEN: 5 * math.pi / 4,
    Component.WHICH:     3 * math.pi / 2,
    Component.HOWMUCH:   7 * math.pi / 4,
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
