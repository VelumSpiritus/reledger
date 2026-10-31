# Reledger

Discovers structural genres from any data and re-files the source
by what the structure actually says.

## Try it

```bash
git clone https://github.com/VelumSpiritus/reledger
cd reledger
pip install numpy
python run.py --download <url>
```

Paste any link. The engine fetches it, detects the data type from
the bytes, and runs. Or point it at a local folder:

```bash
python run.py <folder>
```

The output is a new directory where every source file has been
re-filed into a folder named by the genre the engine discovered.
No labels. No training. No parameters chosen by hand.

## What the engine does

```
bytes → ingest → bridge → (N, 8) → discover → genres → reledger → files by genre
```

1. **Ingest** — fetch, extract, read every file, bridge to 8 features
2. **Discover** — normalize, find the natural genre count, assign
   every file a genre label, report what separates them
3. **Reledger** — re-file source into per-genre folders with a manifest

The bridge is the only data-type-specific piece. Text gets one
bridge, signal data gets another. Both produce the same (N, 8)
shape. Everything after the bridge is identical.

### Text bridge

Every natural language carries a finite, closed class of grammatical
function words — conjunctions, pronouns, determiners, prepositions,
auxiliaries — acquired before any content word and classifiable by
lookup (Talmy, 2000). The engine counts these per window:

```
text → tokenize → classify role → 8 component counts per window → (N, 8)
```

### Signal bridge

```
signal → window → [amplitude, velocity, DC, energy, phase, accel, ZC, spectral] → (N, 8)
```

## The 8 axes

| Component | Grammar (what the bridge reads) | Signal meaning |
|---|---|---|
| WHAT | content nouns, connective conjunctions (and/also), contrastive (but/however) | amplitude |
| HOW | auxiliaries (is/was/can/must), verbs by tense (-ed past, -ing progressive) | velocity |
| WHO | pronouns: subject (I/she), possessive (my/his), object (me/him), reflexive (myself) | DC offset |
| HOWMUCH | quantifiers (many/few/all/some/every), numbers, negator (none) | energy |
| WHEREWHEN | location prepositions (in/on/at), temporal conjunctions (when/before/after) | phase |
| WHY | causal conjunctions (because/therefore/since/hence/so/thus) | acceleration |
| WHETHER | conditionals (if/unless), punctuation (. ! ? , ;), negation (not) | zero-crossings |
| WHICH | selective conjunctions (or/either/neither), interrogative (which) | spectral shape |

The engine reports all 8 after genre discovery — which axes separate
the genres and by how much, with the grammatical grounding for each.

## How genre discovery works

1. **Softmax encoding** — row-wise softmax converts raw features to
   probability distributions
2. **Revolution cycle** — agglomerative clustering by Jensen-Shannon
   divergence, re-classify every window to nearest centroid, repeat
   until the partition converges
3. **Natural-k discovery** — run the revolution cycle at each
   candidate k, pick the k whose converged partition has the highest
   silhouette score

After discovery, the semantic return aggregates each genre's
open-class vocabulary — the content words that were set aside during
classification come back, sorted by which structural genre they
belong to.

## The torus

Each token maps to a point on a flat torus T2:

- **theta_1 (assertion axis):** position in the query cycle —
  opening, establishing, contrasting, closing
- **theta_2 (instrument axis):** which of the 8 components

The flat torus embeds isometrically into R4 via the Clifford torus
(Clifford, 1873):

```
(theta_1, theta_2) -> (cos theta_1, sin theta_1, cos theta_2, sin theta_2) / sqrt(2)
```

Gaussian curvature K = 0. Every point has unit norm.

## Reliability

The grammatical classifier is a deterministic coding scheme on a
closed class. Its reliability is measurable with standard inter-rater
statistics:

- **Cohen's kappa** (Cohen, 1960) — agreement between classifier and
  human coder, corrected for chance. kappa = 1.0 for closed-class
  lookups by construction.
- **Krippendorff's alpha** (Krippendorff, 2004) — multiple coders,
  missing data, ordinal/interval scales.

The open-class heuristics (suffix-based: `-ed`, `-ing`, `-self`,
`'s`) are where disagreement lives and where kappa should be
measured against a human gold standard.

## References

- Clifford, W. K. (1873). Preliminary sketch of biquaternions.
  *Proceedings of the London Mathematical Society*, 4, 381-395.
- Cohen, J. (1960). A coefficient of agreement for nominal scales.
  *Educational and Psychological Measurement*, 20(1), 37-46.
- Cooley, J. W., & Tukey, J. W. (1965). An algorithm for the machine
  calculation of complex Fourier series. *Mathematics of Computation*,
  19(90), 297-301.
- Krippendorff, K. (2004). *Content Analysis: An Introduction to Its
  Methodology* (2nd ed.). Sage.
- Talmy, L. (2000). *Toward a Cognitive Semantics* (Vol. 1). MIT Press.
  Ch. 1: The relation of grammar to cognition — closed-class subsystem.

## License

CC0 1.0 Universal — public domain.
