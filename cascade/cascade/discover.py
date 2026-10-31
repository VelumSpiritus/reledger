"""Cascade level 1: genre discovery.

Normalizes bridged features globally, soft-encodes, finds the natural
genre count via revolution cycle, assigns every window a label.
Calls ingest() at level 0 for the live feature arrays.

CC0 1.0 Universal -- public domain.
"""
import sys
from collections import defaultdict, Counter

import numpy as np

from cascade.ingest import ingest
from lib.engine import (
    COMPONENT_NAMES,
    COMPONENT_MEANINGS,
    COMPONENT_GRAMMAR,
    softmax_batch,
    markov_entropy_rate,
    find_natural_k,
    silhouette_score,
)


def discover(data_dir, temporal=False):
    """Level 1: normalize, discover genres, return live labels + features."""
    sys.stdout.reconfigure(encoding='utf-8')

    print("=" * 70)
    print("RELEDGER — discover structural genres, re-file by what the data says")
    print("=" * 70)
    print()

    print(f"Source: {data_dir}")
    print()

    print("Loading sources...")
    result = ingest(data_dir)
    if result is None:
        print("ERROR: no data loaded")
        return None

    source_names = result["source_names"]
    all_raw = result["all_raw"]
    file_ranges = result["file_ranges"]
    file_vocab = result["file_vocab"]
    file_all_tokens = result["file_all_tokens"]
    file_raw_texts = result["file_raw_texts"]
    vocab = result["vocab"]

    print("\nGlobal normalization...")
    mu, sig = all_raw.mean(0), all_raw.std(0)
    sig[sig < 1e-10] = 1
    normed = (all_raw - mu) / sig
    print(f"  Total windows: {len(normed)}")

    print("Soft encoding (T=0.25)...")
    soft = softmax_batch(normed)
    hard = np.argmax(normed, axis=1)

    profiles = []
    file_sources = []
    file_names = []
    ent_list = []

    for s, e, src, fname in file_ranges:
        profiles.append(soft[s:e].mean(axis=0))
        file_sources.append(src)
        file_names.append(fname)
        ent_list.append(markov_entropy_rate(hard[s:e].tolist()))

    profiles = np.array(profiles)
    features = np.column_stack([profiles, ent_list])

    valid = ~np.isnan(features).any(axis=1) & ~np.isinf(features).any(axis=1)
    if not valid.all():
        n_bad = (~valid).sum()
        features = features[valid]
        file_names = [f for f, v in zip(file_names, valid) if v]
        file_sources = [s for s, v in zip(file_sources, valid) if v]
        print(f"  Dropped {n_bad} files with invalid features")

    n_files = len(features)
    single_source = False
    print(f"  File-level profiles: {n_files}")

    if n_files < 2:
        if len(normed) >= 4:
            print(f"\n  Single source with {len(normed)} observations — clustering rows")
            features = soft
            init_labels = [f"row_{i}" for i in range(len(features))]
            file_sources = init_labels
            file_names = init_labels
            n_files = len(features)
            single_source = True
        else:
            print("\nOnly 1 source — nothing to cluster.")
            print("The engine needs at least 2 files to discover genres.")
            print(f"\n{'=' * 70}")
            print("SUMMARY")
            print(f"{'=' * 70}")
            print(f"  Sources:          {n_files}")
            print("\nProvide a dataset with multiple files, or a directory of documents.")
            return None

    SUBSAMPLE_THRESHOLD = 1000

    print("\nFinding natural genre count...")
    print(f"  {'k':>2s}  {'Revs':>4s}  {'Silhouette':>10s}  Families")
    print("  " + "-" * 60)

    init_labels = file_names if len(set(file_sources)) < 2 else file_sources

    if n_files > SUBSAMPLE_THRESHOLD:
        rng = np.random.RandomState(42)
        sample_idx = rng.choice(n_files, SUBSAMPLE_THRESHOLD, replace=False)
        sample_feats = features[sample_idx]
        print(f"  Subsampling {SUBSAMPLE_THRESHOLD} of {n_files} "
              f"observations for k-discovery...")

        n_init = 30
        centroids_init = sample_feats[
            rng.choice(len(sample_feats), n_init, replace=False)]
        for _ in range(20):
            dists = np.array([[np.linalg.norm(r - c) for c in centroids_init]
                              for r in sample_feats])
            assigns = dists.argmin(axis=1)
            new_c = np.array([sample_feats[assigns == g].mean(axis=0)
                              if (assigns == g).any() else centroids_init[g]
                              for g in range(n_init)])
            if np.allclose(new_c, centroids_init):
                break
            centroids_init = new_c
        pre_labels = [f"g{assigns[i]}" for i in range(len(sample_feats))]

        best_result, all_results = find_natural_k(
            sample_feats, pre_labels)

        for k, sil, labels_k, revs in all_results:
            n_fams = len(set(labels_k))
            print(f"  {k:>2d}  {revs:>4d}  {sil:>10.4f}  {n_fams} families")

        nat_k = best_result[0]
        sample_nat_labels = best_result[2]
        centroids = {}
        for lab in set(sample_nat_labels):
            idxs = [i for i, l in enumerate(sample_nat_labels) if l == lab]
            centroids[lab] = sample_feats[idxs].mean(axis=0)

        print(f"  Assigning all {n_files} observations to {nat_k} families...")
        centroid_arr = np.array(list(centroids.values()))
        centroid_labs = list(centroids.keys())
        all_dists = np.array([[np.linalg.norm(features[i] - c)
                               for c in centroid_arr]
                              for i in range(n_files)])
        nat_labels = [centroid_labs[j] for j in all_dists.argmin(axis=1)]

        sil_n = min(500, n_files)
        sil_idx = rng.choice(n_files, sil_n, replace=False)
        nat_sil = silhouette_score(features[sil_idx],
                                   [nat_labels[i] for i in sil_idx])
        nat_revs = best_result[3]
    else:
        best_result, all_results = find_natural_k(features, init_labels)

        for k, sil, labels_k, revs in all_results:
            n_fams = len(set(labels_k))
            print(f"  {k:>2d}  {revs:>4d}  {sil:>10.4f}  {n_fams} families")

        nat_k, nat_sil, nat_labels, nat_revs = best_result

    print(f"\n  Natural k = {nat_k} (silhouette = {nat_sil:.4f})")

    print(f"\n{'=' * 70}")
    print(f"DISCOVERED FAMILIES (k={nat_k})")
    print(f"{'=' * 70}")

    fam_members = defaultdict(list)
    for i, lab in enumerate(nat_labels):
        fam_members[lab].append((file_sources[i], file_names[i]))

    is_tabular_rows = all(s.startswith("row_") for s in file_sources)
    for fi, (fam_name, members) in enumerate(sorted(fam_members.items())):
        if is_tabular_rows:
            print(f"\n  Family {fi+1}: {len(members)} observations")
        else:
            src_counts = Counter(src for src, _ in members)
            print(f"\n  Family {fi+1}:")
            for src, count in sorted(src_counts.items()):
                print(f"    {src}: {count} files")

    print(f"\n{'=' * 70}")
    print("FAMILY PROFILES")
    print(f"{'=' * 70}")
    print()
    has_entropy = features.shape[1] > 8
    if has_entropy:
        header = "  ".join(f"{c[:5]:>6s}" for c in COMPONENT_NAMES) + f"  {'H':>6s}"
    else:
        header = "  ".join(f"{c[:5]:>6s}" for c in COMPONENT_NAMES)
    print(f"  {'Family':<30s}  {header}")

    for fi, (fam_name, members) in enumerate(sorted(fam_members.items())):
        idxs = [i for i, l in enumerate(nat_labels) if l == fam_name]
        mean_profile = features[idxs].mean(axis=0)
        parts = "  ".join(f"{mean_profile[i]:>6.3f}" for i in range(8))
        if has_entropy:
            print(f"  Family {fi+1:<26d}  {parts}  {mean_profile[8]:>6.4f}")
        else:
            print(f"  Family {fi+1:<26d}  {parts}")

    print(f"\n{'=' * 70}")
    print("WHAT WAS MEASURED — read from the genres")
    print(f"{'=' * 70}")
    print()

    family_profiles = {}
    for fi, (fam_name, members) in enumerate(sorted(fam_members.items())):
        idxs = [i for i, l in enumerate(nat_labels) if l == fam_name]
        family_profiles[fi] = features[idxs].mean(axis=0)

    if len(family_profiles) >= 2:
        fam_arr = np.array([family_profiles[fi]
                            for fi in sorted(family_profiles)])
        comp_variance = fam_arr[:, :8].var(axis=0)
        ent_variance = fam_arr[:, 8].var() if fam_arr.shape[1] > 8 else 0.0
        total_var = comp_variance.sum() + ent_variance

        ranked = sorted(range(8), key=lambda i: comp_variance[i],
                        reverse=True)

        print("  The genres are separated by (most to least):")
        print()
        for rank, ci in enumerate(ranked):
            name = COMPONENT_NAMES[ci]
            pct = (100 * comp_variance[ci] / total_var
                   if total_var > 0 else 0)
            bar = '#' * max(1, int(pct)) if pct >= 0.1 else ''
            print(f"    {rank+1}. {name:<10s}  {pct:>5.1f}%  {bar}")
            print(f"       signal: {COMPONENT_MEANINGS[name]}")
            print(f"       grammar: {COMPONENT_GRAMMAR[name]}")

        ent_pct = 100 * ent_variance / total_var if total_var > 0 else 0
        bar = '#' * max(1, int(ent_pct)) if ent_pct >= 0.1 else ''
        print(f"    9. {'H_rate':<10s}  {ent_pct:>5.1f}%  {bar}")
        print(f"       signal: sequential structure / predictability")
        print(f"       grammar: Markov entropy rate of component transitions")

        print()
        ratio = n_files / nat_k if nat_k > 0 else 0
        unit = "rows" if single_source else "files"
        print(f"  Structural diversity: {nat_k} genres from "
              f"{n_files} {unit} ({ratio:.0f}:1 compression)")
        if ratio > 50:
            print("    High compression — many sources share "
                  "the same grammar.")
            print("    Typical of: natural language, code, "
                  "categorical data.")
        elif ratio > 5:
            print("    Moderate compression — sources cluster "
                  "into distinct families.")
            print("    Typical of: mixed-condition machinery, "
                  "multi-class signals.")
        else:
            print("    Low compression — each source carries "
                  "its own grammar.")
            print("    Typical of: distinct physical sources, "
                  "unique instruments.")

    if vocab:
        print(f"\n{'=' * 70}")
        print("SEMANTIC RETURN — words traced back to source")
        print(f"{'=' * 70}")

        for fi, (fam_name, members) in enumerate(sorted(fam_members.items())):
            idxs = [i for i, l in enumerate(nat_labels) if l == fam_name]
            flat = []
            for idx in idxs:
                if idx >= len(file_vocab):
                    continue
                for w_tokens in file_vocab[idx]:
                    flat.extend(t.lower() for t in w_tokens)

            if not flat:
                continue

            counts = Counter(flat)

            def _trace(word, idxs):
                sources_with = set()
                for idx in idxs:
                    if idx >= len(file_vocab):
                        continue
                    for w_tokens in file_vocab[idx]:
                        if word in [t.lower() for t in w_tokens]:
                            sources_with.add(
                                f"{file_sources[idx]}/{file_names[idx]}")
                            break
                sample = sorted(sources_with)[:3]
                sample_str = ", ".join(sample)
                if len(sources_with) > 3:
                    unit_s = "rows" if single_source else "files"
                    sample_str += f" ... ({len(sources_with)} {unit_s})"
                return sample_str

            top_raw = counts.most_common(15)
            alpha_counts = Counter(
                {w: c for w, c in counts.items()
                 if w.isalpha() and len(w) > 2})
            top_english = alpha_counts.most_common(15)

            unit_s = "rows" if single_source else "files"
            print(f"\n  Family {fi+1} ({len(idxs)} {unit_s}, "
                  f"{len(flat)} content tokens, "
                  f"{len(counts)} unique):")

            print(f"\n    Raw input tokens (what the engine saw):")
            for word, count in top_raw:
                print(f"      {word:<20s} {count:>5d}x  "
                      f"in {_trace(word, idxs)}")

            print(f"\n    English content words:")
            for word, count in top_english:
                print(f"      {word:<20s} {count:>5d}x  "
                      f"in {_trace(word, idxs)}")

    if temporal:
        print(f"\n{'=' * 70}")
        print("DRIFT DETECTION (temporal ordering)")
        print(f"{'=' * 70}")

        for src in source_names:
            src_files = [(i, file_names[i]) for i in range(n_files)
                         if file_sources[i] == src]
            src_files.sort(key=lambda x: x[1])

            if len(src_files) < 10:
                continue

            origin = features[src_files[0][0]]
            drifts = [(fname, np.linalg.norm(features[idx] - origin))
                      for idx, fname in src_files]

            max_drift = max(d for _, d in drifts)
            print(f"\n  {src}: max drift = {max_drift:.4f}")

            ents = [features[idx][8] for idx, _ in src_files]
            baseline_mean = np.mean(ents[:min(50, len(ents) // 5)])
            baseline_std = np.std(ents[:min(50, len(ents) // 5)])
            departure = None
            for si, (idx, fname) in enumerate(
                    src_files[min(50, len(ents) // 5):],
                    start=min(50, len(ents) // 5)):
                if (baseline_std > 0
                        and abs(ents[si] - baseline_mean) > 3 * baseline_std):
                    departure = (si, fname)
                    break
            if departure:
                remaining = len(src_files) - departure[0]
                print(f"    Entropy departure at snapshot "
                      f"{departure[0]}/{len(src_files)} ({departure[1]})")
                print(f"    {remaining} snapshots remaining "
                      f"({100*remaining/len(src_files):.0f}% of life)")
            else:
                print(f"    No entropy departure (stable grammar)")

    component_order = []
    genre_profiles = {}
    if len(family_profiles) >= 2:
        fam_arr = np.array([family_profiles[fi]
                            for fi in sorted(family_profiles)])
        cv = fam_arr[:, :8].var(axis=0)
        component_order = sorted(range(8), key=lambda i: cv[i], reverse=True)
        for fi, (fam_name, _) in enumerate(sorted(fam_members.items())):
            genre_profiles[fam_name] = family_profiles[fi]

    return {
        "nat_labels": nat_labels,
        "nat_k": nat_k,
        "nat_sil": nat_sil,
        "nat_revs": nat_revs,
        "features": features,
        "fam_members": dict(fam_members),
        "file_names": file_names,
        "file_sources": file_sources,
        "file_vocab": file_vocab,
        "file_all_tokens": file_all_tokens,
        "file_raw_texts": file_raw_texts,
        "data_dir": data_dir,
        "source_names": source_names,
        "normed": normed,
        "is_tabular_rows": is_tabular_rows,
        "all_raw": all_raw,
        "vocab": vocab,
        "component_order": component_order,
        "genre_profiles": genre_profiles,
        "single_source": single_source,
    }
