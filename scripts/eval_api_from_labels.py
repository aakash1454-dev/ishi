# scripts/eval_api_from_labels.py
import argparse, time, requests
from pathlib import Path
import pandas as pd

# Optional sklearn metrics
try:
    from sklearn.metrics import roc_auc_score, average_precision_score
except Exception:
    roc_auc_score = average_precision_score = None

def infer(url, path, timeout=20):
    with open(path, "rb") as f:
        r = requests.post(url, files={"image": (Path(path).name, f, "image/jpeg")}, timeout=timeout)
    r.raise_for_status()
    j = r.json()
    return float(j["score"])

def best_threshold(y_true, y_score):
    # grid search for Youden’s J = sens + spec - 1
    grid = [i / 100 for i in range(1, 100)]
    best_j, best_t = -1.0, 0.5
    for t in grid:
        yp = [1 if s >= t else 0 for s in y_score]
        tp = sum(1 for yt, yp_ in zip(y_true, yp) if yt == 1 and yp_ == 1)
        tn = sum(1 for yt, yp_ in zip(y_true, yp) if yt == 0 and yp_ == 0)
        fp = sum(1 for yt, yp_ in zip(y_true, yp) if yt == 0 and yp_ == 1)
        fn = sum(1 for yt, yp_ in zip(y_true, yp) if yt == 1 and yp_ == 0)
        sens = tp / (tp + fn + 1e-9)
        spec = tn / (tn + fp + 1e-9)
        j = sens + spec - 1
        if j > best_j:
            best_j, best_t = j, t
    return best_t

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", default="/workspaces/ishi/datasets/anemia/processed/eyes_defy_anemia/labels.csv")
    parser.add_argument("--url", default="http://localhost:8000/predict")
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--limit", type=int, default=0, help="optional: cap #images for a quick run")
    args = parser.parse_args()

    df = pd.read_csv(args.labels)
    df["image_path"] = df["image_path"].astype(str)
    before = len(df)
    df = df[df["image_path"].str.lower().str.endswith((".jpg", ".jpeg"))].copy()
    print(f"[eval] Using {len(df)}/{before} rows after JPG-only filter")

    df = df[df["label_final"].isin([0.0, 1.0])].copy()
    if args.limit > 0:
        df = df.sample(args.limit, random_state=42).copy()
    df.reset_index(drop=True, inplace=True)

    y_true, y_score = [], []
    pred_rows = []

    t0 = time.time()
    for i, row in df.iterrows():
        p = row["image_path"]
        try:
            s = infer(args.url, p)
        except Exception as e:
            print(f"[skip] {p}: {e}")
            continue

        yt = int(row["label_final"])
        y_true.append(yt)
        y_score.append(s)
        pred_rows.append({
            "image_path": p,
            "country": row.get("country"),
            "subject_id": row.get("subject_id"),
            "label_final": yt,
            "score": float(s),
        })

        if (i + 1) % 50 == 0:
            print(f"{i+1}/{len(df)} ...")

    elapsed = time.time() - t0
    print(f"\nN={len(y_true)}  elapsed={elapsed:.1f}s")

    # Save predictions (only for successfully scored rows)
    pred_df = pd.DataFrame(pred_rows)
    out_csv = "preds_eval.csv"
    pred_df.to_csv(out_csv, index=False)
    print(f"[eval] wrote {out_csv} ({len(pred_df)} rows)")
    if not pred_df.empty:
        print(f"[eval] prevalence (label_final==1): {pred_df['label_final'].mean():.3f}")

    # Global metrics
    if roc_auc_score is not None and len(set(y_true)) == 2:
        try:
            print(f"AUROC={roc_auc_score(y_true, y_score):.4f}")
            print(f"AUPRC={average_precision_score(y_true, y_score):.4f}")
        except Exception:
            pass

    # Threshold diagnostics
    t_star = best_threshold(y_true, y_score) if y_score else args.threshold
    print(f"Best threshold (Youden J) ≈ {t_star:.2f}")

    def dump_conf(t):
        yp = [1 if s >= t else 0 for s in y_score]
        tp = sum(1 for yt, yp_ in zip(y_true, yp) if yt == 1 and yp_ == 1)
        tn = sum(1 for yt, yp_ in zip(y_true, yp) if yt == 0 and yp_ == 0)
        fp = sum(1 for yt, yp_ in zip(y_true, yp) if yt == 0 and yp_ == 1)
        fn = sum(1 for yt, yp_ in zip(y_true, yp) if yt == 1 and yp_ == 0)
        sens = tp / (tp + fn + 1e-9)
        spec = tn / (tn + fp + 1e-9)
        acc  = (tp + tn) / (tp + tn + fp + fn + 1e-9)
        print(f"t={t:.2f}  TP={tp} TN={tn} FP={fp} FN={fn}  Sens={sens:.3f} Spec={spec:.3f} Acc={acc:.3f}")

    dump_conf(args.threshold)
    if abs(t_star - args.threshold) > 1e-6:
        dump_conf(t_star)

    # Per-country diagnostics (only scored rows)
    if not pred_df.empty and roc_auc_score is not None:
        for country, sub in pred_df.groupby("country"):
            yt = sub["label_final"].astype(int).to_numpy()
            ys = sub["score"].astype(float).to_numpy()
            if len(set(yt)) < 2:
                print(f"[eval][{country}] skipped (only one class present)")
                continue
            try:
                auc = roc_auc_score(yt, ys)
                ap  = average_precision_score(yt, ys)
                print(f"[eval][{country}] N={len(sub)} AUROC={auc:.3f} AUPRC={ap:.3f}")
            except Exception as e:
                print(f"[eval][{country}] metrics error: {e}")

if __name__ == "__main__":
    main()
