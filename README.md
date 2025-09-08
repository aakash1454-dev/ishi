# ishi
Non-invasive health screening app
# Anemia Screening (Research Preview)

Files:
- final_predictions_beta.csv
  - columns: filepath, prob_anemic (0-1), decision {non_anemic | review | anemic}
- roc_pr.png, reliability.png (summary plots)
- slice_* CSVs (performance by country/view)
- model: stacked meta + temperature calibration
- thresholding: abstain band [0.30, 0.70]

Intended use:
- Research preview. Not a clinical device.
- “review” means no automatic decision; send to human reviewer.

Operating points:
- Beta mode (abstain band): high confidence calls only; middle band → review.
- Alternate: single threshold selectable to trade off sensitivity vs specificity.

Known limitations:
- Dataset domain: India/Italy palpebral & forniceal_palpebral views.
- Performance varies by view/country slices (see slice CSVs).
- Calibration is approximate; do not treat probabilities as risk scores.
