"""Level 2: re-ledger — reorganize files by discovered genre.

The deepest level of the cascade. Calls discover() which calls
ingest(), receives live labels + features + the engine's component
order, then builds a library whose folder structure follows the
component order the engine generated from the specific byte stream.

CC0 1.0 Universal — public domain.
"""
import os
import sys
import json as json_mod
import shutil

import numpy as np

from cascade.cascade.discover import discover
from lib.engine import COMPONENT_NAMES, find_natural_k, silhouette_score, revolution_cycle
from lib.agreement import cohen_kappa


def reledger_run(data_dir, temporal=False, output_dir=None):
    """The full automatic revolution. One call."""
    result = discover(data_dir, temporal=temporal)
    if result is None:
        return

    nat_labels = result["nat_labels"]
    nat_k = result["nat_k"]
    nat_sil = result["nat_sil"]
    nat_revs = result["nat_revs"]
    features = result["features"]
    fam_members = result["fam_members"]
    file_names = result["file_names"]
    file_sources = result["file_sources"]
    file_raw_texts = result["file_raw_texts"]
    source_names = result["source_names"]
    normed = result["normed"]
    is_tabular_rows = result["is_tabular_rows"]
    all_raw = result["all_raw"]
    n_files = len(features)
    single_source = result.get("single_source", False)
    component_order = result["component_order"]
    genre_profiles = result["genre_profiles"]

    out_base = output_dir or os.path.join(os.path.dirname(data_dir), 'library')

    if os.path.exists(out_base):
        shutil.rmtree(out_base)
    os.makedirs(out_base)

    sorted_fams = sorted(fam_members.items())

    # The engine's component order determines the library structure.
    # Each genre's folder is the component where that genre peaks
    # in the engine's separation ranking. The ranking is reflexive
    # to the byte stream — different data, different order.
    order_rank = {ci: rank for rank, ci in enumerate(component_order)}

    def genre_folder(fam_name):
        if fam_name not in genre_profiles:
            return fam_name
        profile = genre_profiles[fam_name]
        peak = int(np.argmax(profile[:8]))
        return COMPONENT_NAMES[peak].lower()

    # Sort genres by the engine's component order: the genre whose
    # peak falls on the highest-separation axis comes first.
    def sort_key(item):
        fam_name = item[0]
        if fam_name not in genre_profiles:
            return (999,)
        profile = genre_profiles[fam_name]
        peak = int(np.argmax(profile[:8]))
        return (order_rank.get(peak, 999),)

    sorted_fams = sorted(sorted_fams, key=sort_key)

    # Disambiguate genres that peak on the same component
    folder_names = []
    seen = {}
    for fam_name, _ in sorted_fams:
        name = genre_folder(fam_name)
        if name in seen:
            seen[name] += 1
            folder_names.append(f"{name}_{seen[name]}")
        else:
            seen[name] = 1
            folder_names.append(name)
    # Go back and suffix the first occurrence if there were duplicates
    for i, (fam_name, _) in enumerate(sorted_fams):
        base = genre_folder(fam_name)
        if seen[base] > 1 and not folder_names[i].endswith('_1'):
            if folder_names[i] == base:
                folder_names[i] = f"{base}_1"

    print(f"\n{'=' * 70}")
    print("LIBRARY")
    print(f"{'=' * 70}")

    manifest = {
        "source": os.path.abspath(data_dir),
        "genres": nat_k,
        "silhouette": float(nat_sil),
        "revolutions": nat_revs,
        "component_order": [COMPONENT_NAMES[ci].lower()
                            for ci in component_order],
        "families": {},
    }

    for fi, ((fam_name, members), genre_key) in enumerate(
            zip(sorted_fams, folder_names)):
        genre_path = os.path.join(out_base, genre_key)
        os.makedirs(genre_path, exist_ok=True)

        idxs = [i for i, l in enumerate(nat_labels) if l == fam_name]
        family_manifest = {"count": 0, "files": []}

        if is_tabular_rows:
            rows = all_raw[idxs]
            outfile = os.path.join(genre_path, "rows.csv")
            np.savetxt(outfile, rows, delimiter=",",
                       header=",".join(COMPONENT_NAMES), comments="")
            family_manifest["count"] = len(idxs)
            family_manifest["files"].append("rows.csv")
            print(f"  {genre_key}/: {len(idxs)} rows")
        else:
            for src, fname in members:
                src_path = os.path.join(data_dir, src, fname)
                if not os.path.exists(src_path):
                    src_path = os.path.join(data_dir, fname)
                if os.path.exists(src_path):
                    dest_name = f"{src}__{fname}" if src != fname else fname
                    shutil.copy2(src_path,
                                 os.path.join(genre_path, dest_name))
                    family_manifest["files"].append(dest_name)

            genre_text_parts = []
            for idx in idxs:
                if idx < len(file_raw_texts) and file_raw_texts[idx]:
                    genre_text_parts.append(file_raw_texts[idx])
            if genre_text_parts:
                combined_path = os.path.join(genre_path, "combined.txt")
                with open(combined_path, "w", encoding="utf-8") as f:
                    for i, text_part in enumerate(genre_text_parts):
                        if i > 0:
                            f.write("\n\n")
                        f.write(text_part)
                family_manifest["files"].append("combined.txt")

            family_manifest["count"] = len(members)
            unit = "rows" if single_source else "files"
            print(f"  {genre_key}/: {len(members)} {unit}")

        if fam_name in genre_profiles:
            centroid = genre_profiles[fam_name][:8].tolist()
        else:
            centroid = features[idxs].mean(axis=0)[:8].tolist()
        coords = {
            "centroid": {COMPONENT_NAMES[i]: centroid[i]
                         for i in range(len(centroid))},
            "n_windows": len(idxs),
        }
        coords_path = os.path.join(genre_path, "coordinates.json")
        with open(coords_path, "w", encoding="utf-8") as f:
            json_mod.dump(coords, f, indent=2)
        family_manifest["files"].append("coordinates.json")

        manifest["families"][genre_key] = family_manifest

    manifest_path = os.path.join(out_base, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json_mod.dump(manifest, f, indent=2)

    print(f"\n  {os.path.abspath(out_base)}")

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Sources:          {len(source_names)}")
    unit = "Rows" if single_source else "Files"
    print(f"  {unit}:            {n_files}")
    print(f"  Windows:          {len(normed)}")
    print(f"  Natural genres:   {nat_k} (silhouette {nat_sil:.4f})")
    print(f"  Converged in:     {nat_revs} revolutions")
    if temporal:
        print(f"  Drift detection:  enabled")
    print(f"  Library:          {os.path.abspath(out_base)}")
    # -- Reliability: measure genre assignment stability --
    if n_files >= 4 and nat_k >= 2:
        rng = np.random.RandomState(7)
        noise_scale = 0.05 * features.std()
        perturbed = features + rng.randn(*features.shape) * noise_scale
        init_labels_2 = [f"p{i}" for i in range(len(perturbed))]
        nat_labels_2, _ = revolution_cycle(perturbed, init_labels_2, nat_k)

        # Cluster IDs are arbitrary — try every label permutation and
        # take the one that maximises kappa so a relabelled-but-identical
        # partition reads as perfect agreement, not zero.
        from itertools import permutations
        unique_orig = sorted(set(nat_labels))
        unique_pert = sorted(set(nat_labels_2))
        best_kappa, best_pobs, best_pexp = cohen_kappa(nat_labels, nat_labels_2)
        for perm in permutations(unique_orig):
            mapping = dict(zip(unique_pert, perm))
            remapped = [mapping.get(l, l) for l in nat_labels_2]
            k_try, po_try, pe_try = cohen_kappa(nat_labels, remapped)
            if k_try > best_kappa:
                best_kappa, best_pobs, best_pexp = k_try, po_try, pe_try
        kappa, p_obs, p_exp = best_kappa, best_pobs, best_pexp
        manifest["reliability"] = {
            "kappa": round(float(kappa), 4),
            "agreement": round(float(p_obs), 4),
            "method": "perturbed-rerun (5% Gaussian noise)",
        }

        manifest_path = os.path.join(out_base, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json_mod.dump(manifest, f, indent=2)

        print(f"\n{'=' * 70}")
        print("RELIABILITY")
        print(f"{'=' * 70}")
        print(f"  Method:           perturbed rerun (5% Gaussian noise)")
        print(f"  Cohen's kappa:    {kappa:.4f}")
        print(f"  Agreement:        {p_obs:.1%}")
        if kappa >= 0.8:
            print("  Assessment:       substantial — genres are stable under perturbation")
        elif kappa >= 0.6:
            print("  Assessment:       moderate — most genres are stable")
        elif kappa >= 0.4:
            print("  Assessment:       fair — some genre boundaries are sensitive")
        else:
            print("  Assessment:       low — genre assignments shift under small perturbation")

    print()
    print("No labels. No training. No parameters chosen by hand.")
    print("Numbers in, genres out, files re-ledgered by structure.")
