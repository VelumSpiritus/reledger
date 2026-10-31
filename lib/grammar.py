"""Grammatical classification of tokens into interrogative components.

Functional words are a closed class (Talmy, 2000). Every language carries
its interrogative system as a closed class, acquired by every child before
content words. Conjunctions, prepositions, pronouns, determiners,
auxiliaries: all finite, all classifiable by lookup.

Open-class words (nouns, verbs, adjectives, adverbs) are classified by
morphological markers when possible, by position otherwise.
"""

from lib.components import Component, AssertionPhase


# --- Closed-class functional words → components ---

CONJUNCTIONS_CAUSAL = frozenset({
    "because", "since", "therefore", "thus", "hence", "so",
    "consequently", "accordingly", "for",
})

CONJUNCTIONS_CONNECTIVE = frozenset({
    "and", "also", "moreover", "furthermore", "besides",
    "additionally", "plus",
})

CONJUNCTIONS_CONTRASTIVE = frozenset({
    "but", "however", "yet", "although", "though", "whereas",
    "while", "nevertheless", "nonetheless", "despite", "instead",
    "rather", "conversely", "still",
})

CONJUNCTIONS_CONDITIONAL = frozenset({
    "if", "unless", "whether", "provided", "assuming",
    "supposing", "lest",
})

CONJUNCTIONS_TEMPORAL = frozenset({
    "when", "whenever", "while", "before", "after", "until",
    "since", "once", "as",
})

CONJUNCTIONS_SELECTIVE = frozenset({
    "or", "either", "neither", "nor", "else", "otherwise",
})

SUBJECT_PRONOUNS = frozenset({
    "i", "you", "he", "she", "it", "we", "they", "who", "one",
})

OBJECT_PRONOUNS = frozenset({
    "me", "you", "him", "her", "it", "us", "them", "whom",
})

POSSESSIVE_PRONOUNS = frozenset({
    "my", "your", "his", "her", "its", "our", "their", "whose",
    "mine", "yours", "hers", "ours", "theirs",
})

REFLEXIVE_PRONOUNS = frozenset({
    "myself", "yourself", "himself", "herself", "itself",
    "ourselves", "yourselves", "themselves", "oneself",
})

DETERMINERS = frozenset({
    "the", "a", "an", "this", "that", "these", "those",
    "every", "each", "all", "some", "any", "no", "few",
    "many", "much", "several", "both",
})

QUANTIFIERS = frozenset({
    "all", "every", "each", "some", "any", "no", "none",
    "few", "many", "much", "more", "most", "less", "least",
    "several", "enough", "plenty",
})

NEGATORS_BARE = frozenset({
    "not", "barely", "hardly", "scarcely",
})

NEGATORS_WHO = frozenset({
    "nobody",
})

NEGATORS_WHAT = frozenset({
    "no", "nothing",
})

NEGATORS_WHEREWHEN = frozenset({
    "never", "nowhere",
})

NEGATORS_HOWMUCH = frozenset({
    "none",
})

QUESTION_WORDS = frozenset({
    "what", "why", "how", "who", "whom", "whose", "where",
    "when", "which",
})

LOCATION_PREPOSITIONS = frozenset({
    "in", "on", "at", "by", "near", "between", "among",
    "above", "below", "under", "over", "beside", "within",
    "outside", "inside", "behind", "beyond", "through",
    "across", "along", "around", "toward", "towards",
    "from", "into", "onto", "upon",
})

RELATIONAL_PREPOSITIONS = frozenset({
    "of", "with", "for", "about", "than", "as",
    "without", "against", "like", "except", "per",
    "during", "concerning", "regarding",
})

DIRECTIONAL_PREPOSITIONS = frozenset({
    "to", "up", "down", "out", "off", "away",
    "back", "forth", "ahead", "apart",
})

TEMPORAL_ADVERBS = frozenset({
    "now", "then", "already", "still", "yet", "soon",
    "recently", "today", "tomorrow", "yesterday",
    "ago", "later", "early", "late",
})

MANNER_ADVERBS = frozenset({
    "very", "too", "quite", "rather", "just", "even",
    "only", "well", "almost", "also", "perhaps", "often",
    "always", "sometimes", "merely", "simply", "really",
    "actually", "especially", "indeed", "certainly",
})

PLACE_ADVERBS = frozenset({
    "here", "there", "everywhere", "somewhere", "anywhere",
})

AUXILIARIES = frozenset({
    "is", "am", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having",
    "do", "does", "did",
    "will", "would", "shall", "should",
    "can", "could", "may", "might", "must",
})

IRREGULAR_PAST = frozenset({
    "said", "made", "came", "went", "took", "gave", "got",
    "saw", "knew", "thought", "found", "told", "became",
    "left", "felt", "put", "brought", "began", "kept",
    "held", "wrote", "stood", "heard", "let", "meant",
    "set", "met", "ran", "paid", "sat", "spoke", "lay",
    "led", "read", "grew", "lost", "fell", "sent",
    "built", "understood", "rose", "drove", "bought",
    "wore", "chose", "sought", "threw", "broke", "drew",
    "ate", "sang", "woke", "taught", "caught", "spent",
    "won", "hung", "struck", "bound", "cut", "hurt",
    "shut", "hit", "split", "spread", "bid",
})

_CLOSED_CLASS = (
    CONJUNCTIONS_CAUSAL | CONJUNCTIONS_CONNECTIVE | CONJUNCTIONS_CONTRASTIVE
    | CONJUNCTIONS_CONDITIONAL | CONJUNCTIONS_TEMPORAL | CONJUNCTIONS_SELECTIVE
    | SUBJECT_PRONOUNS | OBJECT_PRONOUNS | POSSESSIVE_PRONOUNS | REFLEXIVE_PRONOUNS
    | DETERMINERS | QUANTIFIERS
    | NEGATORS_BARE | NEGATORS_WHO | NEGATORS_WHAT | NEGATORS_WHEREWHEN | NEGATORS_HOWMUCH
    | QUESTION_WORDS | LOCATION_PREPOSITIONS | RELATIONAL_PREPOSITIONS
    | DIRECTIONAL_PREPOSITIONS | TEMPORAL_ADVERBS | MANNER_ADVERBS | PLACE_ADVERBS
    | AUXILIARIES | IRREGULAR_PAST
)

# Past tense markers (English regular verbs)
PAST_SUFFIXES = ("ed", "ied")

# Progressive markers
PROGRESSIVE_SUFFIX = "ing"

# Reflexive suffix
REFLEXIVE_SUFFIXES = ("self", "selves")

# Possessive suffix (open-class: "the dog's bone")
POSSESSIVE_SUFFIX = "'s"


PUNCTUATION_MAP: dict[str, tuple[Component, AssertionPhase]] = {
    ".": (Component.WHETHER, AssertionPhase.CLOSING),
    "!": (Component.WHETHER, AssertionPhase.CLOSING),
    "?": (Component.WHETHER, AssertionPhase.OPENING),
    ",": (Component.WHETHER, AssertionPhase.ESTABLISHING),
    ";": (Component.WHETHER, AssertionPhase.CONTRASTING),
    ":": (Component.WHETHER, AssertionPhase.ESTABLISHING),
    "—": (Component.WHETHER, AssertionPhase.CONTRASTING),
    "–": (Component.WHETHER, AssertionPhase.CONTRASTING),
    "-": (Component.WHETHER, AssertionPhase.CONTRASTING),
    "(": (Component.WHETHER, AssertionPhase.OPENING),
    ")": (Component.WHETHER, AssertionPhase.CLOSING),
    "[": (Component.WHETHER, AssertionPhase.OPENING),
    "]": (Component.WHETHER, AssertionPhase.CLOSING),
    '"': (Component.WHETHER, AssertionPhase.OPENING),
    "'": (Component.WHETHER, AssertionPhase.OPENING),
    "`": (Component.WHETHER, AssertionPhase.OPENING),
}


def classify_token(token: str) -> tuple[Component, AssertionPhase]:
    """Classify a single token by its grammatical role.

    Returns (component, assertion_phase) — the instrument and the position
    in the query cycle.
    """
    if token in PUNCTUATION_MAP:
        return PUNCTUATION_MAP[token]

    t = token.lower().strip(".,;:!?\"'()[]{}—–-")
    if not t:
        return Component.WHETHER, AssertionPhase.OPENING

    # Question words → their named component
    if t in {"what"}:
        return Component.WHAT, AssertionPhase.OPENING
    if t in {"why"}:
        return Component.WHY, AssertionPhase.OPENING
    if t in {"how"}:
        return Component.HOW, AssertionPhase.OPENING
    if t in {"who", "whom", "whose"}:
        return Component.WHO, AssertionPhase.OPENING
    if t in {"where", "when"}:
        return Component.WHEREWHEN, AssertionPhase.OPENING
    if t in {"which"}:
        return Component.WHICH, AssertionPhase.OPENING

    # Causal conjunctions → WHY, establishing direction
    if t in CONJUNCTIONS_CAUSAL:
        return Component.WHY, AssertionPhase.ESTABLISHING

    # Connective conjunctions → WHAT, extending coverage
    if t in CONJUNCTIONS_CONNECTIVE:
        return Component.WHAT, AssertionPhase.ESTABLISHING

    # Contrastive → WHAT, exposing the residual
    if t in CONJUNCTIONS_CONTRASTIVE:
        return Component.WHAT, AssertionPhase.CONTRASTING

    # Conditional → WHETHER (yes/no)
    if t in CONJUNCTIONS_CONDITIONAL:
        return Component.WHETHER, AssertionPhase.OPENING

    # Temporal → WHEREWHEN
    if t in CONJUNCTIONS_TEMPORAL:
        return Component.WHEREWHEN, AssertionPhase.ESTABLISHING

    # Selective → WHICH (resolves underdetermined reference)
    if t in CONJUNCTIONS_SELECTIVE:
        return Component.WHICH, AssertionPhase.OPENING

    # Pronouns → WHO, decomposed by assertion phase:
    # subject opens, possessive establishes,
    # object contrasts, reflexive closes
    if t in REFLEXIVE_PRONOUNS:
        return Component.WHO, AssertionPhase.CLOSING
    if t in SUBJECT_PRONOUNS:
        return Component.WHO, AssertionPhase.OPENING
    if t in POSSESSIVE_PRONOUNS:
        return Component.WHO, AssertionPhase.ESTABLISHING
    if t in OBJECT_PRONOUNS:
        return Component.WHO, AssertionPhase.CONTRASTING

    # Negation decomposed by instrument class, each at CONTRASTING phase
    if t in NEGATORS_BARE:
        return Component.WHETHER, AssertionPhase.CONTRASTING
    if t in NEGATORS_WHO:
        return Component.WHO, AssertionPhase.CONTRASTING
    if t in NEGATORS_WHAT:
        return Component.WHAT, AssertionPhase.CONTRASTING
    if t in NEGATORS_WHEREWHEN:
        return Component.WHEREWHEN, AssertionPhase.CONTRASTING
    if t in NEGATORS_HOWMUCH:
        return Component.HOWMUCH, AssertionPhase.CONTRASTING

    # Quantifiers → HOWMUCH — check before determiners,
    # since the sets overlap and "many/few/all" are quantifiers first
    if t in QUANTIFIERS:
        return Component.HOWMUCH, AssertionPhase.ESTABLISHING

    # Determiners → WHAT (naming the referent)
    if t in DETERMINERS:
        return Component.WHAT, AssertionPhase.ESTABLISHING

    # Location prepositions → WHEREWHEN
    if t in LOCATION_PREPOSITIONS:
        return Component.WHEREWHEN, AssertionPhase.ESTABLISHING

    # Auxiliaries → HOW (process)
    if t in AUXILIARIES:
        return Component.HOW, AssertionPhase.ESTABLISHING

    # Relational prepositions → WHAT (structural relation between covers)
    if t in RELATIONAL_PREPOSITIONS:
        return Component.WHAT, AssertionPhase.ESTABLISHING

    # Directional prepositions → HOW (motion's direction)
    if t in DIRECTIONAL_PREPOSITIONS:
        return Component.HOW, AssertionPhase.OPENING

    # Temporal adverbs → WHEREWHEN
    if t in TEMPORAL_ADVERBS:
        return Component.WHEREWHEN, AssertionPhase.ESTABLISHING

    # Place adverbs → WHEREWHEN
    if t in PLACE_ADVERBS:
        return Component.WHEREWHEN, AssertionPhase.ESTABLISHING

    # Manner/degree adverbs → HOW (manner)
    if t in MANNER_ADVERBS:
        return Component.HOW, AssertionPhase.CONTRASTING

    # Irregular past tense → HOW (completed motion)
    if t in IRREGULAR_PAST:
        return Component.HOW, AssertionPhase.CLOSING

    # --- Open-class heuristics ---

    # Possessive suffix on open-class words ("dog's", "researcher's")
    if t.endswith("'s") and len(t) > 2:
        return Component.WHO, AssertionPhase.ESTABLISHING

    # Reflexive suffix on open-class words (rare but grammatical)
    if any(t.endswith(s) for s in REFLEXIVE_SUFFIXES) and len(t) > 5:
        return Component.WHO, AssertionPhase.CLOSING

    # Past tense → HOW (completed motion)
    if any(t.endswith(s) for s in PAST_SUFFIXES) and len(t) > 3:
        return Component.HOW, AssertionPhase.CLOSING

    # Progressive → HOW (motion in progress)
    if t.endswith(PROGRESSIVE_SUFFIX) and len(t) > 4:
        return Component.HOW, AssertionPhase.ESTABLISHING

    # Adverbs (-ly) → HOW (manner)
    # Guard: "only", "family", "early", "july", "daily", "holy", "assembly",
    # "supply", "rally", "ally", "tally", "belly", "bully", "lily", "fly",
    # "apply", "reply", "rely", "imply", "multiply", "ugly", "lonely"
    # are not manner adverbs — but the guard is length + the suffix appearing
    # AFTER a recognizable stem, not a blocklist. The minimum reliable form
    # is 5+ chars where the stem minus -ly has 3+ chars and isn't a noun root.
    # Simpler: only match -ously, -ively, -ally, -ibly, -ably, -edly, -ingly,
    # -fully, -lessly — compound adverb suffixes with very low false positive rates.
    if any(t.endswith(s) for s in (
        "ously", "ively", "ally", "ibly", "ably", "edly",
        "ingly", "fully", "lessly", "arily", "ately",
    )) and len(t) > 5:
        return Component.HOW, AssertionPhase.CONTRASTING

    # Nominalizations → WHAT at CLOSING (completed cover)
    if any(t.endswith(s) for s in ("tion", "sion", "ment", "ness")) and len(t) > 5:
        return Component.WHAT, AssertionPhase.CLOSING

    # Adjective suffixes → WHAT at CONTRASTING (qualifying the referent)
    # Compound suffixes with near-zero false positive rates:
    # -tive (7+): "active", "creative", "positive" — not "give" (4)
    # -sive (7+): "massive", "passive", "extensive" — not "drive" (5)
    if any(t.endswith(s) for s in ("tive", "sive")) and len(t) > 6:
        return Component.WHAT, AssertionPhase.CONTRASTING
    # -ous (6+): reliable ("serious", "famous")
    # -ful (6+): reliable ("careful", "beautiful")
    # -ical (6+): reliable ("musical", "political")
    if any(t.endswith(s) for s in ("ous", "ful", "ical")) and len(t) > 5:
        return Component.WHAT, AssertionPhase.CONTRASTING
    # -less (7+): excludes "unless" (6)
    # -ible (7+): excludes "bible" (5)
    # -able (7+): excludes "table" (5), "enable" (6)
    if any(t.endswith(s) for s in ("less", "ible", "able")) and len(t) > 6:
        return Component.WHAT, AssertionPhase.CONTRASTING

    # Comparative -er (6+): "bigger", "faster" — not "her" (3), "after" (5)
    if t.endswith("er") and len(t) > 5 and not t.endswith("ter"):
        return Component.WHAT, AssertionPhase.CONTRASTING
    # Superlative -est (6+): "biggest", "fastest" — not "best" (4), "rest" (4)
    if t.endswith("est") and len(t) > 5:
        return Component.WHAT, AssertionPhase.CONTRASTING

    # Third person singular -s on verbs: hard to distinguish from plural nouns
    # without context, so we leave it to the default (WHAT/ESTABLISHING).

    # Numbers → HOWMUCH
    if t.replace(".", "").replace(",", "").isdigit():
        return Component.HOWMUCH, AssertionPhase.ESTABLISHING

    # Default: content words are WHAT (naming the referent)
    return Component.WHAT, AssertionPhase.ESTABLISHING


def is_open_class(token: str) -> bool:
    """True if token is an open-class word (noun, verb, adjective, adverb).

    Closed-class words (pronouns, conjunctions, determiners, prepositions,
    auxiliaries, etc.) return False. Punctuation and numbers return False.
    """
    if token in PUNCTUATION_MAP:
        return False
    t = token.lower().strip(".,;:!?\"'()[]{}—–-")
    if not t:
        return False
    if t.replace(".", "").replace(",", "").isdigit():
        return False
    return t not in _CLOSED_CLASS


def tokenize(text: str) -> list[str]:
    """Split text into tokens. Preserves punctuation as separate tokens."""
    tokens = []
    current = []
    for ch in text:
        if ch.isalnum() or ch == "'":
            current.append(ch)
        else:
            if current:
                tokens.append("".join(current))
                current = []
            if not ch.isspace():
                tokens.append(ch)
    if current:
        tokens.append("".join(current))
    return tokens
