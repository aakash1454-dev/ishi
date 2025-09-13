import re, glob
from pathlib import Path
import pandas as pd

RAW_ROOT = Path("/workspaces/ishi/datasets/anemia/raw/eyes_defy_anemia/dataset_anemia")
OUT_CSV  = Path("/workspaces/ishi/datasets/anemia/processed/eyes_defy_anemia/labels.csv")
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

WHO_MALE   = 13.0
WHO_FEMALE = 12.0

def norm(x): return str(x).strip().lower() if x is not None else ""

ID_KEYS   = ("number","id","subject","subject_id","patient","code","folder","image_id","case","sample")
SEX_KEYS  = ("sex","gender","sesso","genere")
HB_KEYS   = ("hgb","hb","hemoglobin","emoglobina")
FER_KEYS  = ("ferritin","ferritina")
LBL_KEYS  = ("anemia","label","ground","truth","status","class")

def normalize_subject_id(s: str) -> str:
    """Turn '007'/'7.0'/' 7 ' into '7'; leave non-numeric IDs as-is."""
    if s is None: return ""
    t = str(s).strip()
    m = re.fullmatch(r"0*(\d+)(?:\.0*)?", t)
    return m.group(1) if m else t

def pick_col(cols, keys):
    """Find first column containing any key (case-insensitive)."""
    lc = [c.lower() for c in cols]
    for key in keys:
        for i,c in enumerate(lc):
            if key in c:
                return cols[i]
    return None

def load_first_nonempty_sheet(xlsx: Path) -> pd.DataFrame:
    xf = pd.ExcelFile(xlsx, engine="openpyxl")
    for sheet in xf.sheet_names:
        df = pd.read_excel(xlsx, sheet_name=sheet, engine="openpyxl")
        if not df.empty:
            df["_sheet"] = sheet
            return df
    raise RuntimeError(f"No non-empty sheets in {xlsx}")

def build_country(country: str) -> pd.DataFrame:
    xlsx = RAW_ROOT / country / f"{country}.xlsx"
    if not xlsx.is_file():
        return pd.DataFrame()

    df = load_first_nonempty_sheet(xlsx)

    id_col  = pick_col(df.columns, ID_KEYS)
    sex_col = pick_col(df.columns, SEX_KEYS)
    hb_col  = pick_col(df.columns, HB_KEYS)
    fer_col = pick_col(df.columns, FER_KEYS)
    lbl_col = pick_col(df.columns, LBL_KEYS)

    # Make a lookup by normalized subject id
    meta = {}
    for _, row in df.iterrows():
        sid_raw = str(row[id_col]).strip() if id_col else ""
        sid_key = normalize_subject_id(sid_raw)
        if not sid_key: 
            continue
        rec = {"subject_id_raw": sid_raw, "sex": row.get(sex_col) if sex_col else None,
               "hb": None, "ferritin": row.get(fer_col) if fer_col else None,
               "explicit_label": row.get(lbl_col) if lbl_col else None}
        if hb_col:
            hb_val = row.get(hb_col)
            if hb_val is not None and str(hb_val).strip() != "":
                try:
                    rec["hb"] = float(str(hb_val).replace(",", "."))
                except Exception:
                    pass
        meta[sid_key] = rec

    rows = []
    image_dirs = sorted([p for p in (RAW_ROOT / country).glob("[0-9]*") if p.is_dir()])
    for folder in image_dirs:
        sid = folder.name  # e.g. "7"
        key = normalize_subject_id(sid)
        m   = meta.get(key, {})
        imgs = sorted(glob.glob(str(folder / "*.[jJ][pP][gG]"))) + \
               sorted(glob.glob(str(folder / "*.[pP][nN][gG]")))
        for img in imgs:
            hb  = m.get("hb")
            sex = m.get("sex")
            # WHO anemia (if sex known)
            who = None
            if hb is not None:
                s = norm(sex)
                if s in ("m","male"):   who = 1 if hb < WHO_MALE else 0
                elif s in ("f","female"): who = 1 if hb < WHO_FEMALE else 0

            # normalize explicit text labels if present
            explicit = m.get("explicit_label")
            explicit_bin = None
            if explicit is not None:
                se = norm(explicit)
                if se in ("1","anemic","anaemic","positive","yes","y"): explicit_bin = 1
                elif se in ("0","nonanemic","non-anaemic","negative","no","n"): explicit_bin = 0

            final = explicit_bin if explicit_bin is not None else who

            rows.append({
                "image_path": img,
                "country": country,
                "subject_id": sid,
                "sex": sex,
                "hb": hb,
                "ferritin": m.get("ferritin"),
                "label_explicit": explicit_bin,
                "label_who": who,
                "label_final": final,
            })

    return pd.DataFrame(rows)

dfs = []
for country in ("Italy", "India"):
    dfc = build_country(country)
    if not dfc.empty:
        dfs.append(dfc)

all_ = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
all_.to_csv(OUT_CSV, index=False)
print(f"Wrote {OUT_CSV} with {len(all_)} rows")

# Show the exact row for Italy/7/7.jpg if present
sel = all_[all_["image_path"].str.contains("/Italy/7/7.jpg")]
if not sel.empty:
    print("\nRow for Italy/7/7.jpg:")
    print(sel.to_string(index=False))
