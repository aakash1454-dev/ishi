import pandas as pd
import os

df = pd.read_csv('datasets/anemia/processed/eyes_defy_anemia/labels.csv')

print("=== Sample Labels (first 20) ===\n")
for i, row in df.head(20).iterrows():
    path = row['image_path']
    filename = os.path.basename(path)
    label = 'ANEMIC' if row['label_final'] == 1 else 'NOT ANEMIC'
    hb = row['hb']
    subject = row['subject_id']
    sex = row['sex']
    print(f"Subject {subject:3} ({sex}) | Hb={hb:5.1f} | {label:12} | {filename}")

print(f"\n=== WHO Anemia Thresholds ===")
print("Women: Hb < 12.0 g/dL = Anemic")
print("Men:   Hb < 13.0 g/dL = Anemic")

print(f"\n=== Verify a few cases ===")
# Check if labels match WHO thresholds
mismatches = []
for i, row in df.iterrows():
    hb = row['hb']
    sex = row['sex']
    label = row['label_final']
    
    # WHO threshold
    threshold = 12.0 if sex == 'F' else 13.0
    expected = 1 if hb < threshold else 0
    
    if label != expected:
        mismatches.append((row['subject_id'], sex, hb, threshold, label, expected))

print(f"\nLabels that don't match WHO thresholds: {len(mismatches)}")
if mismatches[:5]:
    print("Examples:")
    for m in mismatches[:5]:
        print(f"  Subject {m[0]} ({m[1]}): Hb={m[2]:.1f}, threshold={m[3]}, label={int(m[4])}, expected={m[5]}")

print(f"\n=== Summary ===")
print(f"Total images: {len(df)}")
print(f"Unique subjects: {df['subject_id'].nunique()}")
print(f"Anemic (label=1): {(df['label_final']==1).sum()}")
print(f"Not Anemic (label=0): {(df['label_final']==0).sum()}")
