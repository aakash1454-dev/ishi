import csv, json, subprocess, os, sys, argparse

def run(model, arch, in_csv, out_csv, thresh, img_size, resume=False, flush_every=10, timeout=90):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    # Load existing outputs if resuming
    done = set()
    if resume and os.path.exists(out_csv):
        with open(out_csv, newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                done.add(row["filepath"])

    # Prepare writer
    write_header = True
    if resume and os.path.exists(out_csv):
        write_header = False
        out_f = open(out_csv, "a", newline="")
    else:
        out_f = open(out_csv, "w", newline="")

    fieldnames = ["filepath", "label", "prob_anemic", "pred_label", "arch", "tta_used"]
    w = csv.DictWriter(out_f, fieldnames=fieldnames)
    if write_header:
        w.writeheader(); out_f.flush()

    total = 0
    good = 0
    errs  = 0
    last_flush = 0

    with open(in_csv, newline="") as f:
        rows = list(csv.DictReader(f))

    n = len(rows)
    print(f"[batch_tta] model={model} arch={arch} rows={n} resume={resume}", flush=True)

    for i, row in enumerate(rows, 1):
        img = row.get("filepath") or row.get("path") or row.get("img")
        if not img or not os.path.exists(img):
            errs += 1
            continue
        if img in done:
            continue

        try:
            res = subprocess.run(
                ["python3","scripts/tta_infer.py",
                 "--model", model, "--arch", arch,
                 "--img", img, "--threshold", str(thresh), "--img-size", str(img_size)],
                capture_output=True, text=True, check=True, timeout=timeout
            )
            out = json.loads(res.stdout)
            if "error" in out:
                sys.stderr.write(f"[ERR] {img}: {out['error']}\n")
                errs += 1
            else:
                w.writerow({
                    "filepath": img,
                    "label": row.get("label",""),
                    "prob_anemic": out["prob_anemic"],
                    "pred_label": out["pred_label"],
                    "arch": arch,
                    "tta_used": 1
                })
                good += 1
        except subprocess.TimeoutExpired:
            sys.stderr.write(f"[TIMEOUT] {img}\n"); errs += 1
        except subprocess.CalledProcessError as e:
            sys.stderr.write(f"[ERR] {img}: {e.stderr}\n"); errs += 1
        except Exception as e:
            sys.stderr.write(f"[ERR] {img}: {repr(e)}\n"); errs += 1

        total += 1
        if total - last_flush >= flush_every:
            out_f.flush()
            last_flush = total
            print(f"  [{i}/{n}] written={good} errs={errs}", flush=True)

    out_f.flush(); out_f.close()
    print(f"[batch_tta] DONE written={good} errs={errs} -> {out_csv}", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--arch", required=True)
    ap.add_argument("--in-csv", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--threshold", type=float, default=0.75)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--flush-every", type=int, default=10)
    ap.add_argument("--timeout", type=int, default=90)
    args = ap.parse_args()
    run(args.model, args.arch, args.in_csv, args.out_csv,
        args.threshold, args.img_size, resume=args.resume,
        flush_every=args.flush_every, timeout=args.timeout)
