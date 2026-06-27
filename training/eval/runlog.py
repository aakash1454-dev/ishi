#!/usr/bin/env python3
"""
Per-run provenance: every benchmark run drops a config.json + metrics.json next to
its outputs so a result can always be traced back to (backbone, seed, folds, git
commit, scores). Fixes the current problem that the committed .pth has no origin.

Also the shared metric block for a screening tool: we rank on sensitivity/recall
(catching anemia) and AUC first, F1 second -- a missed case is worse than a false
alarm. anemic = positive class (y = 1).
"""
import json
import pathlib
import subprocess
import time

import numpy as np
from sklearn.metrics import (
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
    average_precision_score,
)


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def screening_metrics(y_true, p_anemic, threshold=0.5):
    """y_true: 0/1 (1=anemic). p_anemic: P(anemic). Returns a metrics dict."""
    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(p_anemic, dtype=float)
    y_pred = (p >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])  # rows=true, cols=pred
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0      # recall on anemic
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    out = {
        "threshold": threshold,
        "n": int(len(y_true)),
        "n_anemic": int(y_true.sum()),
        "sensitivity": float(sens),     # PRIMARY: fraction of anemic caught
        "specificity": float(spec),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    # AUC/AP need both classes present
    if len(np.unique(y_true)) == 2:
        out["auroc"] = float(roc_auc_score(y_true, p))
        out["auprc"] = float(average_precision_score(y_true, p))
    else:
        out["auroc"] = out["auprc"] = None
    return out


def write_run(out_dir, config, metrics, oof=None):
    """Write config.json, metrics.json (+ optional oof_predictions.csv) to out_dir."""
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = {**config, "git_commit": git_commit(), "written_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    if oof is not None:
        import csv
        with open(out_dir / "oof_predictions.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["subject", "y_true", "p_anemic", "fold"])
            for row in oof:
                w.writerow(row)
    return out_dir
