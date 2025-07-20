# ISHI Datasets

This directory contains raw and processed image datasets used for training and testing computer vision models for various health conditions.

---

## 📁 Structure

datasets/
├── anemia/
│ ├── raw/ # Unprocessed images (e.g. from Kaggle, hospital archives)
│ └── processed/ # Resized, labeled, and normalized images
├── other_conditions/
│ ├── cataract/
│ │ ├── raw/
│ │ └── processed/
│ ├── jaundice/
│ │ ├── raw/
│ │ └── processed/
│ └── ... # Add other health conditions here
└── README.md

markdown
Copy
Edit

---

## 🔬 Dataset Guidelines

Each condition-specific dataset should contain:

- `raw/` folder with original, high-resolution images
- `processed/` folder with:
  - Cropped/centered region of interest (e.g. eyelid)
  - Standardized resolution (e.g. 224x224)
  - Normalized pixel values
  - Labels in `labels.csv` or `labels.json`

---

## ✅ Anemia Dataset

- **Source**: Public datasets (e.g. Kaggle) and curated contributions
- **Format**: JPG or PNG
- **Labels**: Binary — `anemic`, `not_anemic`
- **Preprocessing**:
  - Resize to 224×224 pixels
  - Normalize RGB to [0, 1] or standard mean/std
  - Store metadata in `processed/labels.csv`

Example `labels.csv`:

```csv
filename,label
img_0001.jpg,anemic
img_0002.jpg,not_anemic