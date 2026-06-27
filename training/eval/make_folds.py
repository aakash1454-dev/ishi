#!/usr/bin/env python3
"""
Build the ONE shared, leakage-free fold assignment that every model in the
benchmark uses, so comparisons are paired (same subjects in each fold).

  - Grouping unit = subject (country + subject_id). All crops of one person land
    in the same fold. (Today there's one crop per subject, but grouping keeps it
    correct if multiple views are added later.)
  - StratifiedGroupKFold keeps the anemic/nonanemic balance even across folds.
  - Leakage self-check: assert no subject appears in two folds. The split is only
    trusted once this passes -- this is the whole point of the file.

Output: data/folds.csv  (image, country, subject_id, subject, label, y, fold)
where y = 1 for anemic (the positive / clinically-important class), 0 otherwise.

Run:
  uv run python -m training.eval.make_folds          # 5 folds, seed 42
  uv run python -m training.eval.make_folds --folds 5 --seed 42
"""
import argparse
import collections
import csv
import pathlib

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

MANIFEST = pathlib.Path("data/manifest.csv")
OUT = pathlib.Path("data/folds.csv")


def load_manifest(path):
    rows = []
    for r in csv.DictReader(open(path)):
        if r["label"].strip() == "":
            continue
        rows.append(r)
    return rows


def leakage_self_check(rows, n_folds):
    """Hard guarantees the split is trustworthy. Raises on any violation."""
    by_fold = collections.defaultdict(set)
    for r in rows:
        by_fold[int(r["fold"])].add(r["subject"])

    folds = sorted(by_fold)
    assert folds == list(range(n_folds)), f"expected folds 0..{n_folds-1}, got {folds}"

    # (1) no subject shared between any two folds
    for i in folds:
        for j in folds:
            if i < j:
                overlap = by_fold[i] & by_fold[j]
                assert not overlap, f"LEAKAGE: subjects in both fold {i} and {j}: {overlap}"

    # (2) every subject assigned exactly once
    seen = collections.Counter(r["subject"] for r in rows)
    dupes = {s: c for s, c in seen.items() if c > 1}
    # one row per subject today; if multi-crop later, each crop's subject repeats
    # but must all share ONE fold -- check that instead of forbidding repeats
    subj_folds = collections.defaultdict(set)
    for r in rows:
        subj_folds[r["subject"]].add(int(r["fold"]))
    split = {s: f for s, f in subj_folds.items() if len(f) > 1}
    assert not split, f"LEAKAGE: subject(s) split across folds: {split}"

    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(MANIFEST))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = load_manifest(args.manifest)
    for r in rows:
        r["subject"] = f"{r['country']}_{r['subject_id']}"
        r["y"] = 1 if r["label"].strip().lower() == "anemic" else 0

    y = np.array([r["y"] for r in rows])
    groups = np.array([r["subject"] for r in rows])
    X = np.zeros((len(rows), 1))  # unused; StratifiedGroupKFold needs an X

    sgkf = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    fold_of = np.full(len(rows), -1, dtype=int)
    for fold, (_, val_idx) in enumerate(sgkf.split(X, y, groups)):
        fold_of[val_idx] = fold
    assert (fold_of >= 0).all(), "some rows never assigned a fold"
    for r, f in zip(rows, fold_of):
        r["fold"] = int(f)

    leakage_self_check(rows, args.folds)

    rows.sort(key=lambda r: (r["country"], int(r["subject_id"])))
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "country", "subject_id", "subject", "label", "y", "fold"])
        for r in rows:
            w.writerow([r["image"], r["country"], r["subject_id"], r["subject"],
                        r["label"], r["y"], r["fold"]])

    # report: per-fold size + anemic rate (want these roughly equal across folds)
    print(f"leakage self-check PASSED  ({args.folds} folds, seed {args.seed})")
    print(f"{len(rows)} subjects -> {args.out}\n")
    print(f"{'fold':>4} {'n':>4} {'anemic':>7} {'nonanemic':>10} {'anemic%':>8}")
    by_fold = collections.defaultdict(list)
    for r in rows:
        by_fold[r["fold"]].append(r["y"])
    for fold in sorted(by_fold):
        ys = by_fold[fold]
        a = sum(ys); n = len(ys)
        print(f"{fold:>4} {n:>4} {a:>7} {n-a:>10} {100*a/n:>7.1f}%")
    tot = len(rows); ta = sum(r["y"] for r in rows)
    print(f"{'all':>4} {tot:>4} {ta:>7} {tot-ta:>10} {100*ta/tot:>7.1f}%")


if __name__ == "__main__":
    main()
