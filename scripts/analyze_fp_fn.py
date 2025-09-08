# scripts/analyze_fp_fn.py
# Reads a CSV with columns like: filepath, label (0/1), prob_anemic, pred_label
# Adjust column names below if yours differ.

import argparse, os, csv, shutil
from collections import defaultdict

def safe_mkdir(p):
    if not os.path.exists(p):
        os.makedirs(p)

def read_rows(csv_path):
    rows = []
    with open(csv_path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows

def parse_int(x, default):
    try:
        return int(x)
    except:
        return default

def parse_float(x, default):
    try:
        return float(x)
    except:
        return default

def detect_site(path):
    path_low = path.lower()
    if "india" in path_low:
        return "india"
    if "italy" in path_low:
        return "italy"
    return "unknown"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="runs/eval/eval_pipeline_all.csv")
    ap.add_argument("--outdir", default="runs/eval/fp_fn_panels")
    ap.add_argument("--threshold", type=float, default=0.75)
    ap.add_argument("--copy", action="store_true", help="copy images into folders")
    ap.add_argument("--img-col", default="filepath")
    ap.add_argument("--label-col", default="label")         # 0=nonanemic, 1=anemic
    ap.add_argument("--pred-col", default="pred_label")     # optional if present
    ap.add_argument("--prob-col", default="prob_anemic")    # probability of anemic
    args = ap.parse_args()

    rows = read_rows(args.csv)
    total = 0
    tp = fp = tn = fn = 0

    buckets = defaultdict(list)  # keys like "FP_india", "FN_italy"
    site_counts = defaultdict(int)

    for row in rows:
        img = row.get(args.img_col, "")
        label = parse_int(row.get(args.label_col, "0"), 0)
        prob = parse_float(row.get(args.prob_col, "0.0"), 0.0)
        site = detect_site(img)
        site_counts[site] += 1

        pred = 1 if prob >= args.threshold else 0
        total += 1
        if label == 1 and pred == 1:
            tp += 1
        elif label == 0 and pred == 0:
            tn += 1
        elif label == 0 and pred == 1:
            fp += 1
            buckets["FP_" + site].append((img, prob))
        elif label == 1 and pred == 0:
            fn += 1
            buckets["FN_" + site].append((img, prob))

    acc = (tp + tn) / max(total, 1)
    sens = tp / max(tp + fn, 1)  # recall anemic
    spec = tn / max(tn + fp, 1)  # recall non-anemic

    print("Total=%d Acc=%.4f Sens(Anemic)=%.4f Spec(NonAnemic)=%.4f TP=%d FP=%d FN=%d TN=%d" %
          (total, acc, sens, spec, tp, fp, fn, tn))
    print("By site counts:", dict(site_counts))

    # Show top-10 most confident mistakes for quick eyeballing
    for k in sorted(buckets.keys()):
        arr = buckets[k]
        if "FP" in k:
            # High prob wrong as anemic
            arr.sort(key=lambda x: -x[1])
        else:
            # Low prob wrong as non-anemic -> sort ascending
            arr.sort(key=lambda x: x[1])
        print("\n%s (N=%d)" % (k, len(arr)))
        for i, (img, p) in enumerate(arr[:10]):
            print("  %d) p=%.3f  %s" % (i+1, p, img))

    if args.copy:
        for k, arr in buckets.items():
            out_sub = os.path.join(args.outdir, k)
            safe_mkdir(out_sub)
            for img, p in arr:
                base = os.path.basename(img)
                dst = os.path.join(out_sub, "%0.3f__%s" % (p, base))
                try:
                    shutil.copy2(img, dst)
                except Exception as e:
                    pass

if __name__ == "__main__":
    main()
