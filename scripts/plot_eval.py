#!/usr/bin/env python3
import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from pathlib import Path

OUT_DIR = "runs/eval"
RESNET = os.path.join(OUT_DIR, "tta_resnet18_labeled.csv")
MBNET  = os.path.join(OUT_DIR, "tta_mobilenetv3_labeled.csv")
STACK  = os.path.join(OUT_DIR, "stacked_labeled.csv")

os.makedirs(OUT_DIR, exist_ok=True)

def load_probs(path, prob_col, tag):
    df = pd.read_csv(path)
    if prob_col not in df.columns:
        # stacked_labeled.csv has 'prob_stacked'; model files have 'prob_anemic'
        if "prob_stacked" in df.columns and tag=="stack":
            prob_col = "prob_stacked"
        elif "prob_anemic" in df.columns:
            prob_col = "prob_anemic"
        else:
            raise RuntimeError(f"{path} missing prob column (have: {list(df.columns)})")
    y  = df["label"].astype(int).values
    p  = df[prob_col].values.astype(float)
    return y, p

def add_roc_pr(axroc, axpr, y, p, name):
    fpr, tpr, _ = roc_curve(y, p)
    roc_auc = auc(fpr, tpr)
    axroc.plot(fpr, tpr, label=f"{name} AUC={roc_auc:.3f}")

    prec, rec, _ = precision_recall_curve(y, p)
    ap = average_precision_score(y, p)
    axpr.plot(rec, prec, label=f"{name} AP={ap:.3f}")

def reliability(ax, y, p, name, bins=10):
    df = pd.DataFrame({"y":y, "p":p})
    df["bin"] = np.clip((df["p"]*bins).astype(int), 0, bins-1)
    g = df.groupby("bin")
    x = (g["p"].mean()).values
    yhat = (g["y"].mean()).values
    n = g.size().values
    ax.bar(x, yhat, width=1.0/bins, alpha=0.4, align="edge", label=f"{name} (n bins)")
    ax.plot([0,1],[0,1], "--", linewidth=1)
    ax.set_xlim(0,1); ax.set_ylim(0,1)

def main():
    y_r, p_r = load_probs(RESNET, "prob_anemic", "resnet")
    y_m, p_m = load_probs(MBNET,  "prob_anemic", "mbnet")
    y_s, p_s = load_probs(STACK,   "prob_stacked", "stack")

    # shared check (should match)
    assert (y_r == y_m).all() and (y_r == y_s).all(), "Label alignment mismatch across files."

    # ROC + PR
    fig1 = plt.figure(figsize=(10,4))
    ax1  = fig1.add_subplot(1,2,1)
    ax2  = fig1.add_subplot(1,2,2)
    add_roc_pr(ax1, ax2, y_r, p_r, "ResNet18")
    add_roc_pr(ax1, ax2, y_m, p_m, "MobileNetV3")
    add_roc_pr(ax1, ax2, y_s, p_s, "Stacked")
    ax1.plot([0,1],[0,1],'k--',alpha=0.4)
    ax1.set_title("ROC"); ax1.set_xlabel("FPR"); ax1.set_ylabel("TPR"); ax1.legend()
    ax2.set_title("Precision–Recall"); ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision"); ax2.legend()
    fig1.tight_layout()
    rocpr_path = os.path.join(OUT_DIR, "roc_pr.png")
    fig1.savefig(rocpr_path, dpi=150)

    # Reliability (calibration)
    fig2 = plt.figure(figsize=(10,4))
    ax3  = fig2.add_subplot(1,3,1)
    ax4  = fig2.add_subplot(1,3,2)
    ax5  = fig2.add_subplot(1,3,3)
    reliability(ax3, y_r, p_r, "ResNet18")
    ax3.set_title("Reliability – ResNet18")
    reliability(ax4, y_m, p_m, "MobileNetV3")
    ax4.set_title("Reliability – MobileNetV3")
    reliability(ax5, y_s, p_s, "Stacked")
    ax5.set_title("Reliability – Stacked")
    for a in (ax3, ax4, ax5):
        a.set_xlabel("Mean predicted prob (bin)")
        a.set_ylabel("Empirical positive rate")
    fig2.tight_layout()
    rel_path = os.path.join(OUT_DIR, "reliability.png")
    fig2.savefig(rel_path, dpi=150)

    print("Saved:")
    print(" ", rocpr_path)
    print(" ", rel_path)

if __name__ == "__main__":
    main()
