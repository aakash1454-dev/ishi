# EDA Conjunctiva Dataset (masked crops, normalized)

A cleaned set of **218** palpebral-conjunctiva images for anemia detection from the
Eyes-Defy-Anemia (EDA) study, normalized to a single format.

> **Note:** this copy is **EDA-only**. The CP-AnemiC portion of the original combined
> dataset was removed (project focus is EDA). CP-AnemiC is public and re-downloadable
> if external validation is ever needed: CC BY 4.0, DOI `10.17632/m53vz6b7fx.1`.

## Source & attribution
- **Eyes-Defy-Anemia (EDA)** — Dimauro, Maglietta, Bai & Kasiviswanathan.
  Conjunctiva images, Italy + India, adults. IEEE DataPort.
  Cite: Dimauro et al., "An intelligent non-invasive system for automated
  diagnosis of anemia exploiting a novel dataset."

*For local research use. Respect the source license and cite it if you publish.*

## Contents
- `images/<source>/<source>_NNNN.png` — normalized 512x512 **RGBA** crops
  (alpha = conjunctiva mask; flatten onto any background at load time).
- `metadata.csv` — one row per image (218 rows).

| | count |
|---|---|
| Total images | 218 |
| eda-india | 95 |
| eda-italy | 123 |
| Anemic | 91 |
| Non-anemic | 126 |
| Unlabeled (no Hb) | 1 |
| Male / Female | 132 / 86 |

## Normalization applied
- Kept as **RGBA with an alpha mask** of the conjunctiva. Most crops use the
  dataset's native (hand-drawn) masks; the Italy crops saved opaque-on-white get a
  color-key + hole-fill + feathered mask instead.
- Tight-cropped to the mask, aspect-preserving **transparent** pad to square,
  resized to **512x512** (LANCZOS). Flatten onto any background at load time
  (e.g. random backgrounds as augmentation).
- **Color/illumination intentionally left untouched** — pallor is the signal;
  do device/color normalization as a train-time step, not baked into pixels.

## How this differs from `data/crops/`
This folder holds **masked** conjunctiva cut-outs (skin removed via the alpha
channel). `data/crops/` holds **rectangular bounding-box** crops that still include
surrounding skin/eyelid. Keep both — they are the two arms of the
rectangle-vs-masked A/B experiment.

## metadata.csv columns
| column | meaning |
|---|---|
| filename | path to the image, relative to this folder |
| source | eda-india / eda-italy |
| domain | eda |
| subject_id | unique per person; **split by this, never mix across train/test** |
| split | train / val / test (subject-grouped, stratified on `anemic`) |
| anemic | 1 = anemic, 0 = not (target label) |
| hb | hemoglobin, g/dL |
| severity | (empty for EDA) |
| age_years / age_months | age |
| sex | M / F |
| label_provided | (empty for EDA) |
| label_who | anemia derived from Hb via WHO age/sex cutoffs |
| orig_width / orig_height | source crop dimensions before normalization |
| orig_path | original file location |

## Caveats
- **Population confound:** anemia prevalence differs sharply by site
  (India 72% vs Italy 19%), and site is partly inferable from skin tone / camera.
  Always report **within-country** metrics, not just pooled ones.
- Adults only; the two sites differ in capture device and lighting.