from pathlib import Path
import shutil
import os
import csv
from PIL import Image, UnidentifiedImageError

ROOT = Path("/workspaces/ishi")

RAW_DIR = ROOT / 'datasets' / 'anemia' / 'raw' / 'eyes_defy_anemia' / 'dataset_anemia'
PROCESSED_DIR = ROOT / 'datasets' / 'anemia' / 'processed' / 'eyes_defy_anemia'

image_output_dir = PROCESSED_DIR / 'images'
label_csv_path = PROCESSED_DIR / 'labels.csv'

# Create output folders
image_output_dir.mkdir(parents=True, exist_ok=True)

labels = []

def is_valid_image(path):
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except (UnidentifiedImageError, OSError):
        return False

def process_country(country):
    country_dir = RAW_DIR / country
    for subject_id in os.listdir(country_dir):
        subject_path = country_dir / subject_id
        if not subject_path.is_dir():
            continue
        for file in os.listdir(subject_path):
            if file.endswith('_palpebral.png'):
                src = subject_path / file
                if not is_valid_image(src):
                    print(f"⚠️ Skipping invalid image: {src}")
                    continue
                dst = image_output_dir / f"{country}_{subject_id}_{file}"
                shutil.copy(src, dst)
                label = 1 if int(subject_id) % 2 == 0 else 0
                labels.append((dst.name, label))

# Process both India and Italy folders
process_country('India')
process_country('Italy')

# Write labels.csv
with open(label_csv_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['filename', 'label'])
    writer.writerows(labels)

print(f"✅ Copied {len(labels)} valid images to {image_output_dir}")
print(f"✅ Created labels CSV at {label_csv_path}")