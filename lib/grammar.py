"""Grammatical classification of tokens into structural components.

Functional words are a closed class (Talmy, 2000). Every language carries
a finite set of grammatical function words — conjunctions, prepositions,
pronouns, determiners, auxiliaries — acquired before content words and
classifiable by lookup.

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

NEGATORS_C4 = frozenset({
    "nobody",
})

NEGATORS_C1 = frozenset({
    "no", "nothing",
})

NEGATORS_C5 = frozenset({
    "never", "nowhere",
})

NEGATORS_C7 = frozenset({
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
    | NEGATORS_BARE | NEGATORS_C4 | NEGATORS_C1 | NEGATORS_C5 | NEGATORS_C7
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
    ".": (Component.C0, AssertionPhase.CLOSING),
    "!": (Component.C0, AssertionPhase.CLOSING),
    "?": (Component.C0, AssertionPhase.OPENING),
    ",": (Component.C0, AssertionPhase.ESTABLISHING),
    ";": (Component.C0, AssertionPhase.CONTRASTING),
    ":": (Component.C0, AssertionPhase.ESTABLISHING),
    "—": (Component.C0, AssertionPhase.CONTRASTING),
    "–": (Component.C0, AssertionPhase.CONTRASTING),
    "-": (Component.C0, AssertionPhase.CONTRASTING),
    "(": (Component.C0, AssertionPhase.OPENING),
    ")": (Component.C0, AssertionPhase.CLOSING),
    "[": (Component.C0, AssertionPhase.OPENING),
    "]": (Component.C0, AssertionPhase.CLOSING),
    '"': (Component.C0, AssertionPhase.OPENING),
    "'": (Component.C0, AssertionPhase.OPENING),
    "`": (Component.C0, AssertionPhase.OPENING),
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
        return Component.C0, AssertionPhase.OPENING

    # Question words → their named component
    if t in {"what"}:
        return Component.C1, AssertionPhase.OPENING
    if t in {"why"}:
        return Component.C2, AssertionPhase.OPENING
    if t in {"how"}:
        return Component.C3, AssertionPhase.OPENING
    if t in {"who", "whom", "whose"}:
        return Component.C4, AssertionPhase.OPENING
    if t in {"where", "when"}:
        return Component.C5, AssertionPhase.OPENING
    if t in {"which"}:
        return Component.C6, AssertionPhase.OPENING

    # Causal conjunctions → C2, establishing direction
    if t in CONJUNCTIONS_CAUSAL:
        return Component.C2, AssertionPhase.ESTABLISHING

    # Connective conjunctions → C1, extending coverage
    if t in CONJUNCTIONS_CONNECTIVE:
        return Component.C1, AssertionPhase.ESTABLISHING

    # Contrastive → C1, exposing the residual
    if t in CONJUNCTIONS_CONTRASTIVE:
        return Component.C1, AssertionPhase.CONTRASTING

    # Conditional → C0 (binary cut)
    if t in CONJUNCTIONS_CONDITIONAL:
        return Component.C0, AssertionPhase.OPENING

    # Temporal → C5
    if t in CONJUNCTIONS_TEMPORAL:
        return Component.C5, AssertionPhase.ESTABLISHING

    # Selective → C6 (resolves underdetermined reference)
    if t in CONJUNCTIONS_SELECTIVE:
        return Component.C6, AssertionPhase.OPENING

    # Pronouns → C4, decomposed by assertion phase:
    # subject opens, possessive establishes,
    # object contrasts, reflexive closes
    if t in REFLEXIVE_PRONOUNS:
        return Component.C4, AssertionPhase.CLOSING
    if t in SUBJECT_PRONOUNS:
        return Component.C4, AssertionPhase.OPENING
    if t in POSSESSIVE_PRONOUNS:
        return Component.C4, AssertionPhase.ESTABLISHING
    if t in OBJECT_PRONOUNS:
        return Component.C4, AssertionPhase.CONTRASTING

    # Negation decomposed by instrument class, each at CONTRASTING phase
    if t in NEGATORS_BARE:
        return Component.C0, AssertionPhase.CONTRASTING
    if t in NEGATORS_C4:
        return Component.C4, AssertionPhase.CONTRASTING
    if t in NEGATORS_C1:
        return Component.C1, AssertionPhase.CONTRASTING
    if t in NEGATORS_C5:
        return Component.C5, AssertionPhase.CONTRASTING
    if t in NEGATORS_C7:
        return Component.C7, AssertionPhase.CONTRASTING

    # Quantifiers → C7 — check before determiners,
    # since the sets overlap and "many/few/all" are quantifiers first
    if t in QUANTIFIERS:
        return Component.C7, AssertionPhase.ESTABLISHING

    # Determiners → C1 (naming the referent)
    if t in DETERMINERS:
        return Component.C1, AssertionPhase.ESTABLISHING

    # Location prepositions → C5
    if t in LOCATION_PREPOSITIONS:
        return Component.C5, AssertionPhase.ESTABLISHING

    # Auxiliaries → C3 (process)
    if t in AUXILIARIES:
        return Component.C3, AssertionPhase.ESTABLISHING

    # Relational prepositions → C1 (structural relation between covers)
    if t in RELATIONAL_PREPOSITIONS:
        return Component.C1, AssertionPhase.ESTABLISHING

    # Directional prepositions → C3 (motion's direction)
    if t in DIRECTIONAL_PREPOSITIONS:
        return Component.C3, AssertionPhase.OPENING

    # Temporal adverbs → C5
    if t in TEMPORAL_ADVERBS:
        return Component.C5, AssertionPhase.ESTABLISHING

    # Place adverbs → C5
    if t in PLACE_ADVERBS:
        return Component.C5, AssertionPhase.ESTABLISHING

    # Manner/degree adverbs → C3 (manner)
    if t in MANNER_ADVERBS:
        return Component.C3, AssertionPhase.CONTRASTING

    # Irregular past tense → C3 (completed motion)
    if t in IRREGULAR_PAST:
        return Component.C3, AssertionPhase.CLOSING

    # --- Open-class heuristics ---

    # Possessive suffix on open-class words ("dog's", "researcher's")
    if t.endswith("'s") and len(t) > 2:
        return Component.C4, AssertionPhase.ESTABLISHING

    # Reflexive suffix on open-class words (rare but grammatical)
    if any(t.endswith(s) for s in REFLEXIVE_SUFFIXES) and len(t) > 5:
        return Component.C4, AssertionPhase.CLOSING

    # Past tense → C3 (completed motion)
    if any(t.endswith(s) for s in PAST_SUFFIXES) and len(t) > 3:
        return Component.C3, AssertionPhase.CLOSING

    # Progressive → C3 (motion in progress)
    if t.endswith(PROGRESSIVE_SUFFIX) and len(t) > 4:
        return Component.C3, AssertionPhase.ESTABLISHING

    # Adverbs (-ly) → C3 (manner)
    if any(t.endswith(s) for s in (
        "ously", "ively", "ally", "ibly", "ably", "edly",
        "ingly", "fully", "lessly", "arily", "ately",
    )) and len(t) > 5:
        return Component.C3, AssertionPhase.CONTRASTING

    # Nominalizations → C1 at CLOSING (completed cover)
    if any(t.endswith(s) for s in ("tion", "sion", "ment", "ness")) and len(t) > 5:
        return Component.C1, AssertionPhase.CLOSING

    # Adjective suffixes → C1 at CONTRASTING (qualifying the referent)
    if any(t.endswith(s) for s in ("tive", "sive")) and len(t) > 6:
        return Component.C1, AssertionPhase.CONTRASTING
    if any(t.endswith(s) for s in ("ous", "ful", "ical")) and len(t) > 5:
        return Component.C1, AssertionPhase.CONTRASTING
    if any(t.endswith(s) for s in ("less", "ible", "able")) and len(t) > 6:
        return Component.C1, AssertionPhase.CONTRASTING

    # Comparative -er (6+)
    if t.endswith("er") and len(t) > 5 and not t.endswith("ter"):
        return Component.C1, AssertionPhase.CONTRASTING
    # Superlative -est (6+)
    if t.endswith("est") and len(t) > 5:
        return Component.C1, AssertionPhase.CONTRASTING

    # Numbers → C7
    if t.replace(".", "").replace(",", "").isdigit():
        return Component.C7, AssertionPhase.ESTABLISHING

    # Default: content words are C1 (naming the referent)
    return Component.C1, AssertionPhase.ESTABLISHING


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
